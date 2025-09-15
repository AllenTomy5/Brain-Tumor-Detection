# Brain Tumor Detection with CNN

**This project was created for the Hack O Spider Hackathon, under the track Artificial Intelligence and Machine Learning.**

---

## Problem Statement

Early and accurate detection of brain tumors from MRI scans is a critical challenge in healthcare. Manual analysis by radiologists is time-consuming, subject to human error, and often inaccessible in remote or resource-limited settings. There is a pressing need for automated, reliable, and accessible tools that can assist medical professionals in diagnosing brain tumors quickly and accurately, thereby improving patient outcomes and reducing diagnostic delays.

---

## Solution Overview

This project addresses the problem by providing an end-to-end Convolutional Neural Network (CNN) based solution for brain tumor detection from MRI images. The solution includes:

- A robust CNN model trained on a labeled MRI dataset to distinguish between tumor and non-tumor images.
- A user-friendly web application that allows users (doctors, technicians, or patients) to upload MRI images and receive instant predictions, confidence scores, and actionable recommendations.
- Visualizations and statistics to support clinical decision-making.

By automating the detection process, this solution reduces the workload on radiologists, minimizes diagnostic errors, and makes advanced diagnostic support accessible even in under-resourced areas.

---

## Market Scope

The global brain tumor diagnosis and treatment market is rapidly growing, driven by increasing incidence rates and advancements in medical imaging. Automated AI-powered diagnostic tools like this project have significant market potential:

- **Hospitals and Diagnostic Centers:** Can integrate this tool to assist radiologists, reduce turnaround time, and improve diagnostic accuracy.
- **Telemedicine Platforms:** Enables remote diagnosis and second opinions, expanding access to expert care.
- **Emerging Markets:** Provides affordable and scalable diagnostic support where specialist availability is limited.
- **Research and Education:** Useful for training medical students and supporting research in neuro-oncology and medical imaging.

With the rise of AI in healthcare, solutions like this are poised to become essential components of modern diagnostic workflows, improving patient care and operational efficiency.

---

# Brain Tumor Detection with CNN

This project provides a Convolutional Neural Network (CNN) model for brain tumor detection from MRI images, along with a web application for easy usage.

---

## Project Structure

```
accuracy.ipynb
Brain_tumor_detection_CNN.ipynb
Brain_tumor_simple_1.keras
Readme.txt
requirements.txt
Use_model.ipynb
brain_tumor_webapp/
    app.py
    model/
        Brain_tumor_simple_1.keras
    static/
        uploads/
            ...
    templates/
        index.html
Dataset_2/
    test/
        notumor/
        tumor/
    train/
        notumor/
        tumor/
    val/
        notumor/
        tumor/
```

---

## Setup Instructions

1. **Clone the Repository**

   ```sh
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Create and Activate a Virtual Environment**

   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Download/Prepare Dataset**

   - Place your dataset in the `Dataset_2` folder as described above.
   - Ensure the folder structure matches: `train/`, `test/`, `val/` each with `tumor/` and `notumor/` subfolders.

---

## Using the Web Application

1. **Start the Web App**

   ```sh
   cd brain_tumor_webapp
   python app.py
   ```

2. **Access the App**

   - Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)
   - Upload an MRI image to get a prediction.
   - The app will display:
     - Prediction (Tumor detected / No Tumor detected)
     - Confidence score
     - Raw model output
     - Recommended mitigations if a tumor is detected

---

## CNN Model Architecture

The CNN model is implemented in [`Brain_tumor_detection_CNN.ipynb`](Brain_tumor_detection_CNN.ipynb) and consists of the following layers:

- **Input Layer:** Accepts 64x64 RGB images.
- **Convolutional Layer 1:** 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling Layer 1:** 2x2 pool size
- **Convolutional Layer 2:** 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling Layer 2:** 2x2 pool size
- **Flatten Layer:** Converts 2D feature maps to 1D feature vector
- **Dense Layer:** 128 units, ReLU activation
- **Dropout Layer:** 0.5 dropout rate to prevent overfitting
- **Output Layer:** 1 unit, sigmoid activation (for binary classification)

**Model Compilation:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy

**Training:**
- The model is trained on the MRI dataset with data augmentation using Keras' `ImageDataGenerator`.

**Saving:**
- The trained model is saved as `Brain_tumor_simple_1.keras` and used in both the notebook and the web application.

---

## Notebooks

- [`Brain_tumor_detection_CNN.ipynb`](Brain_tumor_detection_CNN.ipynb): Model training and evaluation.
- [`accuracy.ipynb`](accuracy.ipynb): Testing model accuracy on the dataset.
- [`Use_model.ipynb`](Use_model.ipynb): Simple script to load the model and predict on a single image.

---

## Notes

- Make sure to adjust file paths as needed for your environment.
- For best results, use high-quality MRI images.
- The web app stores uploaded images temporarily in `brain_tumor_webapp/static/uploads/`.

---
## Tech Stack

- **Python 3**: Core programming language for model development and backend.
- **TensorFlow / Keras**: Deep learning framework for building and training the CNN model.
- **NumPy**: Numerical operations and data manipulation.
- **Matplotlib & Seaborn**: Data visualization and plotting in notebooks.
- **Pillow**: Image processing and manipulation.
- **Flask**: Lightweight web framework for serving the web application.
- **Werkzeug**: WSGI utility library used by Flask.
- **Jupyter Notebook**: Interactive development and experimentation environment.
- **HTML/CSS/JavaScript**: Frontend for the web application (with custom dark mode styling).