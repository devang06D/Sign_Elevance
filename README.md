# Indian Sign Language (ISL) Detection System

A deep learning-based Indian Sign Language recognition system that can detect alphabets (A-Z) and numbers (0-9) from hand gestures using a Convolutional Neural Network (CNN).

---

## 🎯 Project Objective

The main goal of this project is to develop an intelligent system that can recognize **Indian Sign Language (ISL)** gestures in real-time and from static images. This helps bridge the communication gap between hearing-impaired individuals and the general public by translating sign language into readable text.

**Key Features:**
- Detects 36 classes (Digits 0-9 + Letters A-Z)
- User-friendly GUI with image upload functionality
- Displays prediction with confidence score in a separate result window
- Built using TensorFlow/Keras and OpenCV
- Suitable for accessibility and educational purposes

---

## 📁 Dataset

- **Source**: Custom ISL Dataset
- **Total Classes**: 36 (0-9 and A-Z)
- **Structure**:
  - `train/` → Contains training images for each class
  - `test/`  → Contains testing images for each class
- Each class folder contains hundreds of images of correct hand gestures (verified manually)

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras** (Deep Learning)
- **OpenCV** (Image Processing)
- **Tkinter** (GUI)
- **NumPy & PIL** (Image handling)
- **Matplotlib** (Training visualization)

---

## 📊 Model Details

- **Architecture**: Custom CNN with 3 Convolutional blocks
- **Input Size**: 100x100 RGB images
- **Output Classes**: 36
- **Training**: Trained on custom ISL dataset with proper train/test split
- **Framework**: TensorFlow 2.x

---

## Project Structure
```
Sign-Language-Detection/
├── isl_dataset/                      # main dataset
│   ├── train/                        # Training images 
│   └── test/                         # Testing images 
├── models/
│   └── isl_sign_language_model_36classes.h5   # Your trained model
├── newTrain.py                       # Training script (you named it)
├── newgui.py                         # GUI script (you named it)


```

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ISL-Sign-Language-Detection.git
cd ISL-Sign-Language-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training (Optional - if you want to retrain)
```bash
python newTrain.py
```

### 4. Run the GUI Application
```bash
python newgui.py
```

#### To see the models output Click here : https://drive.google.com/drive/folders/1KnnUYA8Mvl6yUy24fIIpmNYUM63yoet9?usp=sharing
