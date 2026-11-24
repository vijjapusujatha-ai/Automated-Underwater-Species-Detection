# 🐠 Automated Species Identification in Underwater Videos Using YOLOv8

## 📌 Project Overview

Automated Species Identification in Underwater Videos Using YOLOv8 is a computer vision project designed to identify marine species in underwater video footage using the YOLOv8 object detection model.

The project uses a trained YOLOv8 model to detect objects in underwater videos and presents the detection results through a Flask-based web application.

## 🎯 Objectives

- Identify marine species in underwater videos.
- Apply YOLOv8 for object detection.
- Display detected objects with bounding boxes.
- Provide a simple web interface for video-based detection.

## ✨ Key Features

- Underwater video upload.
- YOLOv8-based object detection.
- Trained model integration (`best.pt`).
- Flask web application.
- Visual presentation of detection results.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| YOLOv8 | Object detection |
| OpenCV | Video processing |
| Flask | Web application |
| PyTorch | Deep learning framework |
| HTML/CSS | User interface |

## 📂 Project Structure

```text
Automated-Underwater-Species-Detection/
│
└── UnderWater/
    └── UnderWater/
        ├── app.py
        ├── app_backup.py
        ├── best.pt
        ├── requirements.txt
        ├── README.md
        ├── Underwater__Species.ipynb
        ├── test_camera.html
        ├── templates/
        │   ├── home.html
        │   ├── index.html
        │   └── detection.html
        └── uploads/
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vijjapusujatha-ai/Automated-Underwater-Species-Detection.git
```

### 2. Navigate to the application folder

```bash
cd Automated-Underwater-Species-Detection/UnderWater/UnderWater
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment (Windows)

```powershell
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

### 7. Open in your browser

```text
http://127.0.0.1:5000
```

## 🧠 Model Information

- Model: YOLOv8
- Model file: `best.pt`
- Task: Object detection
- Input: Underwater video footage

## 🚀 Future Enhancements

- Improve detection accuracy in low-visibility underwater environments.
- Support real-time underwater species detection.
- Expand the dataset with additional marine species.
- Improve model performance and processing speed.
- Deploy the application online.

## 👩‍💻 Project Team

**Project Title:** Automated Species Identification in Underwater Videos Using YOLOv8

## 📌 GitHub Repository

[Automated Underwater Species Detection](https://github.com/vijjapusujatha-ai/Automated-Underwater-Species-Detection)

---

**Developed as a Final Year B.Tech Project.**