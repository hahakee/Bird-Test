from ultralytics import YOLO

# Load a model
model = YOLO('./runs/classify/train3/weights/best.pt')  # load an official model
#model = YOLO("path/to/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.top1  # top1 accuracy
metrics.top5  # top5 accuracy