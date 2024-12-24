from ultralytics import YOLO
import numpy as np
# Load a model
model = YOLO('./runs/classify/train3/weights/best.pt')  # load an official model

# Predict with the model
results = model('D:/BirdTest/ultralytics-main/ultralytics-main\images1/train/002.Laysan_Albatross/Laysan_Albatross_0003_1033.jpg',show=True)  # predict on an image

names_dict = results[0].names

probs = results[0].probs.data.tolist()

print(names_dict)

# 打印概率
print(probs)

# 打印最高概率对应的类别
print(names_dict[np.argmax(probs)])