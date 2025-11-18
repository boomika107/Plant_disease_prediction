# Plant_disease_prediction
# ShreyaML: Plant Disease Classification and Severity Estimation

## Overview

ShreyaML is a web application built with Flask that provides real-time classification of plant diseases from uploaded images and estimates the severity of the infection. It leverages a hybrid machine learning backend, utilizing both TensorFlow/Keras and PyTorch models for robust analysis.

The application supports the identification of 42 different classes across major crops including Cotton, Wheat, Maize, Rice, and Sugarcane.

## Features

*   **Dual-Model Prediction:** Uses a Keras model (`models/model1.keras`) for primary disease classification and a PyTorch model (`models/model2.pth`) for severity estimation.
*   **Interactive Results:** Displays model confidence and estimated severity using interactive Plotly gauge charts.
*   **Disease Definitions:** Provides local definitions, summaries, and sources for identified diseases from `data/definitions.json`.
*   **Care Guides:** Offers tailored care tips based on the predicted disease and severity level.
*   **Web Interface:** Simple, user-friendly interface for image upload and result display.

## Technology Stack

*   **Backend:** Python, Flask
*   **Machine Learning:** TensorFlow/Keras, PyTorch, Torchvision, NumPy
*   **Data Visualization:** Plotly
*   **Image Processing:** Pillow (PIL)
*   **Frontend:** HTML, CSS

## Installation and Setup

### Prerequisites

Ensure you have Python 3.8+ installed.

### 1. Clone the Repository

```bash
git clone <repository_url>
cd shreyaML
```

### 2. Set up Virtual Environment

It is highly recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies

Install the required Python packages. Note that this project requires both TensorFlow and PyTorch, as listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Model Placement

The application requires two pre-trained models to function correctly. If these models are missing, the application will run in a simulated mode using random predictions (as seen in `app.py`).

Please place your trained models in the following paths:

*   **Classification Model:** `models/model1.keras` (TensorFlow/Keras format)
*   **Severity Model:** `models/model2.pth` (PyTorch weights/state dict)

## Usage

### Running the Application

Start the Flask development server:

```bash
python app.py
```

The application will typically be accessible at `http://127.0.0.1:5000/`.

### Making Predictions

1.  Open the application URL in your web browser.
2.  Use the file upload form to select an image of a plant leaf or crop part.
3.  Submit the image to receive the classification result, confidence score, severity estimate, and recommended care guide.

## Data

The `data/definitions.json` file contains structured information (title, summary, source) for all supported disease classes, which is used to enrich the prediction results.

## Contributing

If you wish to contribute, please ensure all dependencies are met and models are correctly placed. Follow standard Python and Flask best practices.
