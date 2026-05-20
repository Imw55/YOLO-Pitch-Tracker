import torch
from ultralytics import YOLO
import os

def train(md_name, ds_path, epochs=100, batch_sz=16, img_sz = 640, lr=0.01):

    device = 'cpu'
    if torch.cuda.is_available(): device = '0'

    model = YOLO('yolo12n.pt')

    model.train (
        data=os.path.join(ds_path,"data.yaml"),
        epochs=epochs,
        batch=batch_sz,
        imgsz=img_sz,
        patience=10,
        save=True,
        device=device,
        project=os.path.join(r"C:\Github\CEG4195_Project","runs"),
        name=md_name,
        lr0=lr,
        lrf=0.01,
        plots=True,
        save_period=5
    )

    return model

model = train("TV2", r"C:\Github\CEG4195_Project\DatasetV2_Reduced")
