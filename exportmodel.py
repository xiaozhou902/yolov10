from ultralytics import YOLOv10

#Load a model

model = YOLOv10('yolov10n.pt') # load an official model
model = YOLOv10('/root/yolov10-main/runs/detect/train64/weights/best.pt') # load a custom trained model
#Export the model
model.export(format='onnx')