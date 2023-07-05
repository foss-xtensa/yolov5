import argparse

from save_yolo_pt import create_yolo4d_model
from load_pretrained import load_pretrained_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a new YOLOv5 model with only 4D tensors")
    parser.add_argument('-c','--cfg_file', help='Path to YOLOv5 config file', default='models/yolov5m.yaml', required=True)
    parser.add_argument('-w','--weights', help='Path to pretrained yolo model weights in .pt', default='yolov5m.pt', required=True)
    parser.add_argument('-s','--save_file', help='Path to save the new yolo pt model', default='yolov5m-4d.pt' ,required=True)

    args = parser.parse_args()

    create_yolo4d_model(args.cfg_file, args.save_file)
    load_pretrained_weights(args.save_file, args.weights, args.save_file)