# Kolam AI 🎨✨

An AI-powered web application to classify, generate, and learn traditional Kolam art in Augmented Reality. This project combines a Python backend with a WebXR frontend to bring intricate Kolam designs to life.

*An illustration of the Kolam AI application running on a mobile device, showing a Kolam design projected in AR.*

-----

## \#\# Features

  * **AR Kolam Visualization**: Use WebXR to project and view Kolam designs in your own space.
  * **AI-Powered Classification**: Upload a Kolam image to classify its type (e.g., Pulli, Sikku, Kambi) using a deep learning model.
  * **Procedural Generation**: Generate new Kolam designs based on type and complexity.
  * **Interactive Drawing Tutorial**: Learn to draw a Kolam step-by-step with animated guides in both 3D and AR.
  * **Image-to-Vector Pipeline**: An advanced OpenCV script that analyzes a Kolam image, skeletonizes its lines, and converts it into a smooth vector format.

-----

## \#\# Tech Stack

  * **Backend**: **Python**, **Flask**, **PyTorch**, **OpenCV**, **Scipy**
  * **Frontend**: **HTML5**, **CSS3**, **JavaScript**, **Three.js**, **WebXR**

-----

## \#\# Project Structure

```
kolam-ai-app/
├── app.py                  # Main Flask application
├── train.py                # PyTorch script for model training
├── vectorizer.py           # OpenCV pipeline for Kolam analysis
├── requirements.txt        # Backend dependencies
├── kolam_dataset/          # Folder for training/validation images
│   ├── train/
│   └── val/
├── static/
│   ├── script.js           # Frontend Three.js & WebXR logic
│   ├── style.css           # Stylesheet
│   ├── uploads/            # Directory for user-uploaded images
│   └── generated/          # Directory for generated Kolam SVGs
└── templates/
    └── index.html          # Main HTML page
```

-----

## \#\# Setup and Installation

Follow these steps to get the project running locally.

### \#\#\# Prerequisites

  * Python 3.8+
  * A mobile device with AR capabilities (Android with ARCore) for AR features.

### \#\#\# 1. Clone the Repository

```bash
git clone https://github.com/your-username/kolam-ai-app.git
cd kolam-ai-app
```

### \#\#\# 2. Create `requirements.txt`

Create a file named `requirements.txt` with the following content:

```
flask
torch
torchvision
opencv-contrib-python
numpy
scipy
```

### \#\#\# 3. Set Up a Python Virtual Environment

It is highly recommended to use a virtual environment.

```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### \#\#\# 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### \#\#\# 5. Run the Application

```bash
flask run
```

The application will be running at `http://127.0.0.1:5000`.

-----

## \#\# Usage

### \#\#\# Desktop Mode

Open a web browser and navigate to `http://127.0.0.1:5000`. You can interact with the Kolam in a 3D viewer, generate new designs, and test the API endpoints.

### \#\#\# AR Mode

1.  Ensure your computer and mobile device are on the **same Wi-Fi network**.
2.  Find your computer's local IP address (e.g., `192.168.1.10`).
3.  Open Chrome on your mobile device and navigate to `http://<YOUR_IP_ADDRESS>:5000`.
4.  Tap the **"Enter AR"** button, scan a flat surface, and tap to place the Kolam.

-----

## \#\# Modules Overview

  * **`app.py`**: The core Flask server. It serves the frontend and provides API endpoints for `/classify`, `/generate`, and `/api/teach`.
  * **`train.py`**: A PyTorch script for training the Kolam classification model using transfer learning on a pretrained ResNet. Requires a dataset in the `kolam_dataset` folder.
  * **`vectorizer.py`**: A standalone computer vision script using OpenCV and Scipy to analyze a Kolam image, trace its lines and dots, and convert them into a smooth, vectorized format.
  * **`static/script.js`**: The heart of the frontend. This file contains all the Three.js logic for 3D/AR rendering, API communication, and handling user interactions.

-----

## \#\# Contributing

Contributions are welcome\! If you have ideas for improvements or new features, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Commit your changes (`git commit -m 'Add some amazing feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

-----

## \#\# License

This project is licensed under the MIT License. See the `LICENSE` file for details.
