import torch
from ultralytics import YOLO
import os

def train(md_name, ds_path, epochs=30, img_sz = 640, lr=0.0005):

    device = 'cpu'
    if torch.cuda.is_available():
        device = '0'

    model = YOLO(r"C:\Github\CEG4195_Project\best.pt")

    model.train(
        data=os.path.join(ds_path, "data.yaml"),
        epochs=epochs,
        imgsz=img_sz,

        device=device,

        project=r"C:\Github\CEG4195_Project\runs",
        name=md_name,

        patience=0,
        lr0=lr,
        freeze=10,

        plots=True,

        scale=0.9,
        translate=0.2,
        mosaic=1.0,
        multi_scale=True,

        save=True,
        save_period=5
    )

    return model

model = train("TV3", r"C:\Github\CEG4195_Project\Dataset_Scale")
