import argparse
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import RANK


# Build a custom YOLO model
def load_model(self, cfg=None, weights=None, verbose=True):
  model = RTDETRDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
  weights, _ = attempt_load_one_weight("checkpoints/rtdetr-x.pt")
  model.load(weights)
  return model




if __name__ == '__main__':
    # Another helmet dataset: https://osf.io/4pwj8/
    parser = argparse.ArgumentParser(description="rtdetr helmet experiment")
    parser.add_argument('-config', type=str, default="helmet_data.yaml",  help="config file for model training")
    parser.add_argument('-devices', type=int, default=1,  help="number of gpus")
    parser.add_argument('-model', type=str, default="rtdetr-x.yaml", help="model file name")
    parser.add_argument('-epochs',   type=int, default=10,  help="number of epoch")
    parser.add_argument('-bs',      type=int, default=16, help="number of batches")
    parser.add_argument('-imgsz',   type=int, default=640, help="resize image before feeding to model")
    parser.add_argument('-rpath',   type=str, default="/home/results/", help="path to results")
    parser.add_argument('-name',    type=str,   default="rtdetr-x-increase-augment-all", help="run name")
    parser.add_argument('-project', type=str,   default="helmets-challenge", help="project name")
    parser.add_argument('-frac',    type=float, default=1.0, help="fraction of the data being used")
    args = parser.parse_args()

    
    RTDETRTrainer.get_model = load_model

    device = 0 if args.devices == 1 else [i for i in range(args.devices)]

    train_args = dict(project=args.project, 
                      name=args.name,
                      model=args.model, 
                      data=args.config,
                      device=device, 
                      epochs=args.epochs, 
                      batch=args.bs, 
                      fraction=args.frac, 
                      imgsz=args.imgsz,
                      exist_ok=True,
                      val=True, 
                      #freeze=10,
                      #save_json=True, 
                      conf=0.001, 
                      iou=0.5,
                      #lr0=0.001,
                      #optimizer="AdamW", 
                      #seed=0,
                      #box=7.5, 
                      #cls=0.125, 
                      #dfl=3.0,
                      #close_mosaic=0,
                      hsv_h=0.5, # (float) image HSV-Hue augmentation (fraction)
                      hsv_s=0.8, # (float) image HSV-Saturation augmentation (fraction)
                      hsv_v=0.8, # (float) image HSV-Value augmentation (fraction)
                      degrees=10.0, # (float) image rotation (+/- deg)
                      translate=0.1, # (float) image translation (+/- fraction)
                      scale=0.8, # (float) image scale (+/- gain)
                      shear=0.0, # (float) image shear (+/- deg)
                      perspective=0.0001, # (float) image perspective (+/- fraction), range 0-0.001
                      flipud=0.8, # (float) image flip up-down (probability)
                      fliplr=0.5, # (float) image flip left-right (probability)
                      mosaic=1, # (float) image mosaic (probability)
                      mixup=0.0, # (float) image mixup (probability)
                      copy_paste=0.0, # (float) segment copy-paste (probability)
                      auto_augment="randaugment", # (str) auto augmentation policy for classification (randaugment, autoaugment, augmix)
                      erasing=0.6, # (float) probability of random erasing during classification training (0-1)
                      )

    trainer = RTDETRTrainer(overrides=train_args)
    #trainer.add_callback("on_val_end", save_eval_json_with_id)
    trainer.train()