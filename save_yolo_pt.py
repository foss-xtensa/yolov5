import argparse

import torch

from models.yolo import Model

"""
Command Line to export the model to ONNX from .pt
python export.py --weights yolov5-4d.pt --include onnx
"""

def create_yolo4d_model(cfg_file='models/yolov5m.yaml', save_file='yolov5-4d.pt'):
    print(f'Loading YOLOv5 config from : {cfg_file}')
    model = Model(cfg_file, ch=3, nc=80)
    ckpt = {'model':model}
    torch.save(ckpt, save_file)
    print(f'Saved YOLOv5 Model to : {save_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a new YOLOv5 model with only 4D tensors")
    parser.add_argument('-c','--cfg_file', help='Path to YOLOv5 config file', default='models/yolov5m.yaml', required=False)
    parser.add_argument('-s','--save_file', help='Path to save the new yolo pt model', default='yolov5-4d.pt' ,required=False)
    args = parser.parse_args()

    create_yolo4d_model(args.cfg_file, args.save_file)