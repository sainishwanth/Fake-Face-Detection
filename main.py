import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow import keras
import numpy as np

loaded_model = None
global_image = None

def load_image():
    global global_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")])
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((label.winfo_width(), label.winfo_height()))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo
        global_image = file_path
        run_button.pack()

def load_model():
    global loaded_model
    loaded_model = tf_load_model("image_model.h5")
    if loaded_model:
        print("Model loaded successfully!")

def run_model():
    global loaded_model, global_image
    if loaded_model:
        image = Image.open(global_image)
        image = image.resize((224, 224))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        prediction = loaded_model.predict(image)
        if prediction[0][0] == 0:
            messagebox.showinfo("Prediction", "Fake")
        elif prediction[0][0] == 1:
            messagebox.showinfo("Predictio", "Real")
        else:
            messagebox.showerror("Error", "Error")
    else:
        messagebox.showwarning("Warning", "No model loaded. Please load a model first.")

root = tk.Tk()
root.title("Image Loader and Model Runner")

window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")

load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

label = tk.Label(root)
label.pack(expand=True, fill="both")

load_model_button = tk.Button(root, text="Load Model", command=load_model)
load_model_button.pack(pady=10)

run_button = tk.Button(root, text="Run Model", command=run_model)
run_button.pack_forget()

root.mainloop()
