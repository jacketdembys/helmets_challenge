#!/usr/bin/python3
"""
Evaluate submissions for the Detecting Violation of Helmet Rule for Motorcyclists track of the AI City Challenge.
Track 5 in 2024.
"""
import os
import traceback
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = ArgumentParser(add_help=False, usage=usageMsg())
    parser.add_argument("--predictions_file", type=str, help="path to predictions file", required=True)
    parser.add_argument("--ground_truth_file", type=str, help="path to ground truth file", required=True)
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    parser.add_argument('--width', type=int, default=1920, help="Video width in pixels.")
    parser.add_argument('--height', type=int, default=1080, help="Video height in pixels.")
    parser.add_argument('-s', '--static_bbox', type=str, default="static_bbox.txt", help="Static bounding boxes.")
    parser.add_argument('-o', '--occlusion_bbox', type=str, default="occlusion_bbox.txt", help="Occlusion bounding boxes.")

    return parser.parse_args()


def usageMsg():
    return """  python3 aicityeval-helmet.py --ground_truth_file <ground_truth> --predictions_file <prediction> -o <occlusion bboxes file> -s <static bboxes file> [--height <video height> --width <video width>]

Details for expected formats for each track can be found at https://www.aicitychallenge.org/.

See `python3 aicityeval-helmet.py --help` for more info.
"""


def getData(fpath, names=['vid', 'fid', 'xmin', 'ymin', 'width', 'height', 'cid', 'conf'], sep='\s+|\t+|,'):
    """ Get the necessary track data from a file handle.

    Params
    ------
    fh : opened handle
        Steam handle to read from.
    fpath : str
        Original path of file reading from.
    names : list<str>
        List of column names for the data.
    sep : str
        Allowed separators regular expression string.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    """

    try:
 
        df = pd.read_csv(
            fpath,
            sep=sep,
            index_col=None,
            header=None,
            engine='python'
        )
        df.columns = names[:len(df.columns)]
        df['xmax'] = df.xmin + df.width
        df['ymax'] = df.ymin + df.height
        return df

    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (os.path.basename(fpath), repr(e)))


def validate(x):
    """
    Validate uploaded data.
    Args:
        x: data frame containing predictions
    """
    if x.cid.min() < 1 or x.cid.max() > 9:
        raise ValueError('Class id is out of range. It should be between 1 and 7 and be an integer.')
    if x.vid.min() < 1 or x.vid.max() > 100:
        raise ValueError('Video id is out of range. It should be between 1 and 100 and be an integer.')
    if x.fid.min() < 1 or x.fid.max() > 200:
        raise ValueError('Frame id is out of range. It should be between 1 and 200 and be an integer.')
    if np.any((x.xmin < 0) | (x.xmin > 1920)):
        raise ValueError('Xmin value is out of range. It should be between 0 and 1920.')
    if np.any((x.width <= 0) | (x.width > 1920)):
        raise ValueError('Width value is out of range. It should be between 1 and 1920.')
    if np.any((x.ymin < 0) | (x.ymin > 1080)):
        raise ValueError('Ymin value is out of range. It should be between 0 and 1080.')
    if np.any((x.height <= 0) | (x.height > 1080)):
        raise ValueError('Height value is out of range. It should be between 1 and 1080.')
    if 'conf' not in x.columns:
        raise ValueError('The confidence score is missing.')
    if x.conf.min() < 0.0 or x.conf.max() > 1.0:
        raise ValueError('Confidence is out of range. It should be between 0 and 1.')


def overlap1(p, q):
    """
    Args
        p: pandas series with xmin, ymin, xmax, ymax columns
        q: pandas series with xmin, ymin, xmax, ymax columns
    Returns
        overlap: overlap score between p and q
    """
    area = (q.xmax - q.xmin) * (q.ymax - q.ymin)
    iw = max(min(p.xmax, q.xmax) - max(p.xmin, q.xmin), 0)
    ih = max(min(p.ymax, q.ymax) - max(p.ymin, q.ymin), 0)
    ua = (p.xmax - p.xmin) * (p.ymax - p.ymin) + area - iw * ih
    return (iw * ih / ua) if ua != 0 else 0.

def overlap(gt, q):
    """
    Args
        gt: data frame with xmin, ymin, xmax, ymax columns
        q:  pandas series with xmin, ymin, xmax, ymax columns
    Returns
        overlaps: pandas series with overlaps between the query and each sample in gt
    """
    area = (q.xmax - q.xmin) * (q.ymax - q.ymin)
    iw = (gt.xmax.clip(upper=q.xmax) - gt.xmin.clip(lower=q.xmin)).clip(lower=0.)
    ih = (gt.ymax.clip(upper=q.ymax) - gt.ymin.clip(lower=q.ymin)).clip(lower=0.)
    ua = (gt.xmax - gt.xmin) * (gt.ymax - gt.ymin) + area - iw * ih
    return (iw * ih / ua).fillna(0.)

def max_overlap(gt, qs):
    """
    Get maximum overlap of each query among all gt boxes
    """
    return pd.concat([overlap(gt, q) for _,q in qs.iterrows()], axis=1).max(axis=1)

def prfilter(pr, oc, rb, ocpct=0.9, rbpct=0.05):
    """
    Filter out predictions that have the same class type and an overlap of at least 0.9 with
    pre-defined bounding boxes for the given frame. A class matches if both the occlusion and
    predicted bounding boxes belong to the driver (cid 1) or the same passenger (cid 2 & 3 for 
    passenger 1, 4 & 5 for passenger 2, etc.).
    Also filter out bounding boxes that are too small (with < 40 or height < 40) or that overlap
    redaction bounding boxes by at least 5%.
    Args:
        pr: data frame containing predictions
        oc: data frame containing occlusions
        rb: data frame containing redactions
        ocpct: minimum occlusion overlap to trigger filter
        rbpct: minimum redaction overlap to trigger filter
    """
    dropids = []
    for i, r in pr.iterrows():
        if r.width < 40 or r.height < 40:
            dropids.append(i)
            continue
        filtered = False
        for _, s in oc[(oc.vid == r.vid) & (oc.fid == r.fid)].iterrows():
            if r.cid//2 == s.cid//2 and overlap1(r, s) >= ocpct:
                dropids.append(i)
                filtered = True
                break
        if filtered:
            continue
        for _, b in rb[(rb.vid == r.vid) & (rb.fid == r.fid)].iterrows():
            if overlap1(r, b) >= rbpct:
                dropids.append(i)
                break

    return pr.drop(index=dropids)


def ap(tp, m):
    """
    Compute Average Precision given the tp and fp arrays
    Args: 
        tp : numpy.Array    Array with 1s for true-positives
    """
    # cumulate tp and fp
    fp = np.zeros(tp.size)
    for i in range(tp.size):
        if tp[i] == 0:
            fp[i] = 1
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    # compute precision and recall
    rec = tp / m
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # append sentinel values at start and end of prec and rec
    mpre = np.concatenate(([0.], prec, [0.]))
    mrec = np.concatenate(([0.], rec, [1.]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # find points where recall changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # finally, sum (\Delta recall) * prec
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def compute_map(gt, pr):
    """
    Compute the mean average precision for the predictions in `pr` given the ground
    truth in `gt` for both the full data set and the 50% subset of the test set
    containing only the videos with ids in `half_vids`.
    Args:
        gt : pandas.DataFrame   Ground truth data
        pr : pandas.DataFrame   Predictions data
    Return:
        hmap, fmap
    """
    gt['selected'] = False
    pr = pr.sort_values(by='conf', ascending=False)  # sort in non-increasing confidence order
    pr['tp'] = 0
    fmaps = []  # mean average precision scores for the full test set
    for cid in sorted(gt.cid.unique()):  # compute average precision for each class
        cgt = gt[gt.cid == cid]
        cpr = pr[pr.cid == cid]
        m = len(cgt)  # number of gt with this class
        # process data one video and one frame at a time

        for vid in sorted(cpr.vid.unique()):
            cvpr = cpr[cpr.vid == vid]
            for fid in sorted(cvpr.fid.unique()):
                cvfpr = cvpr[cvpr.fid == fid]
                cvfgt = cgt[(cgt.vid == vid) & (cgt.fid == fid)]
                
                if cvfgt.empty:
                    continue
                for _, r in cvfpr.iterrows():
                    cvfgt['iou'] = overlap(cvfgt, r)  # overlap (IOU) of r against all gt bounding boxes in the frame
                    s = cvfgt[cvfgt.iou == cvfgt.iou.max()].iloc[0]
                    if s.iou >= 0.5 and not cgt.loc[s.name, 'selected']:
                        cgt.loc[s.name, 'selected'] = True
                        cpr.loc[r.name, 'tp'] = 1
        
        fmaps.append(ap(cpr.tp.values, m))  # compute AP for full test set
        
    return np.mean(fmaps)


def usage(msg=None):
    """ Print usage information, including an optional message, and exit. """
    if msg:
        print("%s\n" % msg)
    print("\nUsage: %s" % usageMsg())
    exit()



if __name__ == '__main__':
    args = get_args()

    try:
        pr = getData(args.predictions_file)  # read predictions
        validate(pr)  # validate the predictions
        gt = getData(args.ground_truth_file)  # read ground truth
        oc = getData(args.occlusion_bbox)  # read occlusions
        rb = getData(args.static_bbox)  # read redactions boxes
        pr = prfilter(pr, oc, rb)  # filter predictions
        mp = compute_map(gt, pr)  # evaluate the predictions
        print('MAP: %s\n' % mp)
    except Exception as e:
            print("Error: %s" % str(e))
            traceback.print_exc()


