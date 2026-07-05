# 🌿 Plant Disease Recognition System

A deep learning-based web application that identifies plant diseases from leaf images using a Convolutional Neural Network (CNN), deployed as an interactive Streamlit app.

![Plant Disease](home_page.jpeg)

## Overview

This system classifies leaf images into **38 categories** spanning healthy and diseased states across multiple crop species (Apple, Corn, Grape, Potato, Tomato, and more). Users upload a leaf image through the web interface and receive an instant disease prediction.

## Dataset

- **Source:** PlantVillage dataset (recreated via offline augmentation from the original dataset)
- **Classes:** 38 categories of healthy and diseased crop leaves
- **Split:**
  - Train: 70,295 images
  - Validation: 17,572 images
  - Test: 33 images

## Model Architecture

A custom CNN built from scratch in TensorFlow/Keras using a `Sequential` architecture:

- 5 convolutional blocks with increasing filter depth (32 → 64 → 128 → 256 → 512), each using two `Conv2D` layers followed by `MaxPooling2D`
- `Dropout` layers (0.25 and 0.4) to reduce overfitting
- Fully connected `Dense` layer (1500 units, ReLU)
- Output `Dense` layer (38 units, softmax) for multi-class classification

**Training configuration:**
- Input size: 128×128×3
- Optimizer: Adam (learning rate = 0.0001)
- Loss: Categorical Crossentropy
- Epochs: 10
- Batch size: 32

## Results

Trained over 10 epochs, the model achieved:
- **Training accuracy:** ~97%
- **Validation accuracy:** ~95%

Training history (loss/accuracy per epoch) is logged in `training_hist.json` for reproducibility.

## Web Application

Built with **Streamlit**, the app has three pages:

1. **Home** — Introduction to the system
2. **About** — Dataset details and class breakdown
3. **Disease Recognition** — Upload a leaf image and get a real-time prediction

The app loads the trained model (`trained_model.keras`) and returns the predicted disease class along with the uploaded image preview.

## Tech Stack

- **Deep Learning:** TensorFlow, Keras, CNN
- **Web Framework:** Streamlit
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **ML Utilities:** scikit-learn

## Project Structure

```
├── Train_plant_disease.ipynb   # Model training notebook
├── Test_Plant_Disease.ipynb    # Model evaluation/testing notebook
├── main.py                     # Streamlit application
├── trained_model.keras         # Saved trained model
├── training_hist.json          # Training history (loss/accuracy per epoch)
├── requirement.txt             # Python dependencies
└── home_page.jpeg              # Home page banner image
```

## Setup & Usage

1. Clone the repository
   ```bash
   git clone <https://github.com/NilabhSharma/Plant-Disease-Prediction-System>
   cd plant-disease-recognition
   ```

2. Install dependencies
   ```bash
   pip install -r requirement.txt
   ```

3. Run the app
   ```bash
   streamlit run main.py
   ```

4. Open the local URL shown in the terminal, upload a leaf image on the **Disease Recognition** page, and click **Predict**.

## Future Improvements

- Add data augmentation (rotation, flipping, zoom) to improve generalization on unseen images
- Experiment with transfer learning (e.g., MobileNetV2, EfficientNet) to compare performance against the custom CNN
- Expand test set for more robust evaluation
- Add confidence scores alongside predictions in the UI
