import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
from datetime import datetime

MODEL_PATH = "isl_sign_language_model_36classes.h5" 
IMG_SIZE = (100, 100)

if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Error", f"Model file '{MODEL_PATH}' not found!\nTrain the model first.")
    exit()

model = load_model(MODEL_PATH, compile=False)
print("model loaded successfully")

class_folders = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]
label_to_class = {i: name for i, name in enumerate(class_folders)}

root = tk.Tk()
root.title("ISL Sign Language Detector")
root.geometry("900x700")
root.configure(bg="#f8fafc")
root.resizable(False, False)

current_image_array = None
img_label_widget = None

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        messagebox.showerror("Error", f"Image processing failed: {e}")
        return None

def show_result_window(sign_name, confidence, class_idx):
    result_win = tk.Toplevel(root)
    result_win.title("Detection Result")
    result_win.geometry("640x460")
    result_win.configure(bg="#f8fafc")
    
    # Center the result window
    x = (result_win.winfo_screenwidth() - 640) // 2
    y = (result_win.winfo_screenheight() - 460) // 2
    result_win.geometry(f"640x460+{x}+{y}")

    tk.Label(result_win, text="Detection Result", font=("Arial", 22, "bold"), 
             bg="#f8fafc", fg="#1e40af").pack(pady=30)

    tk.Label(result_win, text=f"Class Index: {class_idx}", font=("Arial", 11), 
             bg="#f8fafc", fg="#64748b").pack()

    color = "#22c55e" if confidence > 70 else "#eab308" if confidence > 50 else "#ef4444"
    
    tk.Label(result_win, text=sign_name, font=("Arial", 80, "bold"), 
             fg=color, bg="#f8fafc").pack(pady=20)

    tk.Label(result_win, text=f"Confidence: {confidence:.2f}%", 
             font=("Arial", 18), bg="#f8fafc", fg="#334155").pack(pady=15)

    tk.Button(result_win, text="Close", font=("Arial", 12, "bold"), width=18, height=2,
              bg="#64748b", fg="white", command=result_win.destroy).pack(pady=40)

def predict_and_show():
    global current_image_array
    if current_image_array is None:
        messagebox.showwarning("Warning", "Please upload an image first!")
        return

    try:
        predictions = model.predict(current_image_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        sign_name = label_to_class.get(class_idx, f"Unknown ({class_idx})")

        show_result_window(sign_name, confidence, class_idx)
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

def upload_image():
    global current_image_array, img_label_widget
    
    file_path = filedialog.askopenfilename(
        title="Select Sign Language Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    try:
        pil_img = Image.open(file_path).resize((220, 220))
        tk_img = ImageTk.PhotoImage(pil_img)

        if img_label_widget is None:
            img_label_widget = tk.Label(image_frame, image=tk_img, bg="white", relief="solid", bd=3)
            img_label_widget.pack(pady=25)
        else:
            img_label_widget.config(image=tk_img)
        
        img_label_widget.image = tk_img
        current_image_array = preprocess_image(file_path)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image:\n{e}")


tk.Label(root, text="Indian Sign Language Detector", 
         font=("Arial", 26, "bold"), bg="#f8fafc", fg="#1e40af").pack(pady=25)

image_frame = tk.Frame(root, bg="#f8fafc")
image_frame.pack(pady=10)

tk.Label(image_frame, text="Image will be displayed here", 
         font=("Arial", 12), fg="#64748b", bg="#f8fafc").pack(pady=30)

btn_frame = tk.Frame(root, bg="#f8fafc")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Upload Image", font=("Arial", 13, "bold"), 
          width=18, height=2, bg="#22c55e", fg="white", command=upload_image).grid(row=0, column=0, padx=45)

tk.Button(btn_frame, text="Detect Sign", font=("Arial", 13, "bold"), 
          width=18, height=2, bg="#3b82f6", fg="white", command=predict_and_show).grid(row=0, column=1, padx=45)

tk.Label(root, text="Elevance Internship Project", 
         font=("Arial", 10), bg="#f8fafc", fg="#64748b").pack(side="bottom", pady=25)

root.mainloop()