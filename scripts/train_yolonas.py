import argparse
import wandb
import os
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val
)
from super_gradients.training import Trainer

wandb.init(project="helmets-challenge")


if __name__ == '__main__':
    # Another helmet dataset: https://osf.io/4pwj8/
    parser = argparse.ArgumentParser(description="yolov8x helmet experiment")
    parser.add_argument('-config', type=str, default="helmet_data.yaml",  help="config file for model training")
    parser.add_argument('-devices', type=int, default=1,  help="number of gpus")
    parser.add_argument('-epochs',   type=int, default=10,  help="number of epoch")
    parser.add_argument('-bs',      type=int, default=16, help="number of batches")
    parser.add_argument('-imgsz',   type=int, default=640, help="resize image before feeding to model")
    parser.add_argument('-rpath',   type=str, default="/home/results/", help="path to results")
    parser.add_argument('-name',    type=str,   default="yolov8x", help="run name")
    parser.add_argument('-project', type=str,   default="helmets-challenge", help="project name")
    parser.add_argument('-frac',    type=float, default=1.0, help="fraction of the data being used")
    args = parser.parse_args()

    
    # Import the trainer 
    trainer = Trainer(
        experiment_name='transfer_learning_object_detection_yolo_nas', 
        ckpt_root_dir="./checkpoints/"
    )

    # Set up the dataloader
    labels = []
    dataset_params = {
        'data_dir': '/home/dataset',
        'train_images_dir':'train/images',
        'train_labels_dir':'train/labels',
        'val_images_dir':'train//images',
        'val_labels_dir':'train//labels',
        'test_images_dir':'train//images',
        'test_labels_dir':'train//labels',
        'classes': labels
    }



    # Initialize weights and bias
    # wandb.init(
    #    entity="jacketdembys",
    #    project = "ai-city-challenge",                
    #    group = 'Test-0-Playing-With-The-API',
    #    name = "Test-Inference",
    #    job_type = "baseline"
    # )

    # Load a model and export to onnx format: the model is visualize in the Netron app: https://netron.app
    #yolo_v8 = "x"   # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x 
    #model = YOLO("yolov8"+yolo_v8+".pt")  # load a model

    device = 0 if args.devices == 1 else [i for i in range(args.devices)]

    train_args = dict(project=args.project, 
                      name=args.name,
                      model="yolov8x.yaml", 
                      data=args.config,
                      device=device, 
                      epochs=args.epochs, 
                      batch=args.bs, 
                      fraction=args.frac, 
                      imgsz=args.imgsz,
                      exist_ok=True,
                      val=True, 
                      #save_json=True, 
                      #conf=0.5, 
                      #iou=0.5,
                      #optimizer="Adam", 
                      #seed=0,
                      #box=7.5, 
                      #cls=0.125, 
                      #dfl=3.0,
                      #close_mosaic=0,
                      #degrees=0.1, 
                      #translate=0.1, 
                      #scale=0.0, 
                      #shear=0.0, 
                      #perspective=0.0, 
                      #flipud=0.0, 
                      #fliplr=0.5, 
                      #mosaic=0.0, 
                      #mixup=0.0
                      )

    trainer = DetectionTrainer(overrides=train_args)
    #trainer.add_callback("on_val_end", save_eval_json_with_id)
    trainer.train()