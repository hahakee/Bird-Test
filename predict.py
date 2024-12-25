import numpy as np
import tkinter as tk
from tkinter import Toplevel, Label
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from ultralytics import YOLO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 导入 FigureCanvasTkAgg

# Load a model
model = YOLO('./runs/classify/train3/weights/best.pt')  # load an official model

# Predict with the model
results = model('D:/BirdTest/ultralytics-main/ultralytics-main/images1/train/002.Laysan_Albatross/Laysan_Albatross_0003_1033.jpg', show=True)  # predict on an image

# Get class names and probabilities
names_dict = results[0].names
probs = results[0].probs.data.tolist()

# Create a tkinter window to display the results
root = tk.Tk()
root.title("Prediction Results")

# Open the image and convert it to a format tkinter can display
img_path = 'D:/BirdTest/ultralytics-main/ultralytics-main/images1/train/002.Laysan_Albatross/Laysan_Albatross_0003_1033.jpg'
img = Image.open(img_path)
img.thumbnail((300, 300))  # Resize the image for better display
img_tk = ImageTk.PhotoImage(img)

# Create a Label widget to show the image
img_label = Label(root, image=img_tk)
img_label.pack()

# Create a new window to show the text results
result_window = Toplevel(root)
result_window.title("Prediction Details")

# Display class names and probabilities
output_text = ""
for i, prob in enumerate(probs):
    output_text += f"{names_dict[i]}: {prob:.2f}\n"

# Get the highest probability and corresponding class
max_prob_class = names_dict[np.argmax(probs)]
output_text += f"\nHighest Probability Class: {max_prob_class}"

# Create a label to show the output text
result_label = Label(result_window, text=output_text, font=('Helvetica', 12), padx=10, pady=10)
result_label.pack()

# Create a histogram of the probabilities
fig, ax = plt.subplots()
ax.bar(names_dict.values(), probs)
ax.set_xlabel('Class')
ax.set_ylabel('Probability')
ax.set_title('Class Probabilities')

# Show the histogram in a new window
hist_window = Toplevel(root)
hist_window.title("Probability Histogram")

# Display the histogram using a Canvas widget
canvas = FigureCanvasTkAgg(fig, hist_window)
canvas.get_tk_widget().pack()

# Start the tkinter main loop
root.mainloop()
