import wandb
import argparse
import os
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback


if __name__ == '__main__':
    # Another helmet dataset: https://osf.io/4pwj8/
    parser = argparse.ArgumentParser(description="yolov8x helmet experiment")
    parser.add_argument('-devices', type=int, default=1,  help="number of gpus")
    parser.add_argument('-epochs',   type=int, default=10,  help="number of epoch")
    parser.add_argument('-bs',      type=int, default=16, help="number of batches")
    parser.add_argument('-imgsz',   type=int, default=640, help="resize image before feeding to model")
    parser.add_argument('-rpath',   type=str, default="/home/retina/dembysj/Dropbox/WCCI2024/challenges/ETSS-01-Edge-TSS/src/aic23/track_5/", help="path to results")
    args = parser.parse_args()

    # Initialize weights and bias
    wandb.init(
        entity="jacketdembys",
        project = "ai-city-challenge",                
        group = 'Test-0-Playing-With-The-API',
        name = "Test-Inference",
        job_type = "baseline"
    )

    # Load a model and export to onnx format: the model is visualize in the Netron app: https://netron.app
    yolo_v8 = "x"   # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x 
    model = YOLO("yolov8"+yolo_v8+".pt")  # load a model

    device = 0 if args.devices == 1 else [i for i in range(args.devices)]
    model.to(device)

    # Visualize on Wandb the model
    add_wandb_callback(model)

    # Train the model
    results = model.train(data="helmet_data.yaml", 
                        device=device,
                        epochs=args.epochs,
                        batch=args.bs,
                        imgsz=args.imgsz,
                        project=os.path.join(args.rpath, "results"),
                        verbose=True)