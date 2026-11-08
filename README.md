# 🚗 AI-Powered Driver Monitoring System

An intelligent, real-time Driver Monitoring System (DMS) built using **Python, OpenCV, and Dlib**. This project leverages computer vision techniques to detect driver drowsiness and distraction, helping improve road safety by providing timely visual and audible alerts.

The system continuously monitors the driver's facial landmarks, eye movements, and head orientation through a webcam to identify potentially unsafe driving behavior.

## ✨ Key Features

### 👁️ Real-Time Drowsiness Detection
- Monitors the driver's Eye Aspect Ratio (EAR).
- Detects prolonged eye closure, which may indicate drowsiness or microsleep.
- Triggers an alert when the eye closure duration exceeds a predefined threshold.

### 🧠 Real-Time Distraction Detection
- Uses Head Pose Estimation to analyze the driver's head orientation.
- Detects when the driver looks away from the road.
- Identifies potential distractions in different directions, including:
  - Looking Left
  - Looking Right
  - Looking Down

### 🔊 Audible & Visual Alerts
- Displays real-time warning messages on the video feed.
- Plays an audible alarm when drowsiness or distraction is detected.
- Supports a looping, stoppable alarm to alert the driver effectively.

### 🖥️ Real-Time Video Feed
- Displays a live webcam feed.
- Visualizes facial landmarks and eye contours.
- Provides annotations indicating the driver's head direction.
- Offers real-time visual feedback for monitoring driver behavior.

## 🎬 Demo

A demonstration of the application running in real time.

<!-- Replace the placeholder below with your actual GIF or video -->

![Driver Monitoring System Demo](assets/demo.gif)

> **Note:** Replace `assets/demo.gif` with the path to your actual demonstration GIF. You can create a short GIF showing drowsiness detection, distraction detection, and the alert system in action.

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.x | Core programming language |
| OpenCV | Real-time computer vision, video capture, and image processing |
| Dlib | Face detection and facial landmark prediction |
| NumPy | Numerical operations and vector/matrix calculations |
| SciPy | Euclidean distance calculations between facial landmarks |
| Pygame | Audio processing and audible alert management |

## ⚙️ System Requirements

Before installing the project, ensure that you have the following:

- Python 3.x
- A working webcam
- A microphone is not required
- A compatible operating system (Windows, Linux, or macOS)
- Internet connection for downloading dependencies

## 📥 Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/AjaxOnTheWay/DMS_project.git
```

Navigate to the project directory:

```bash
cd DMS_project
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install opencv-python dlib numpy scipy pygame
```

> **Note:** Dlib may require CMake and a compatible C++ compiler, depending on your operating system and Python version.

### 4. Download the Facial Landmark Model

The project requires a pretrained facial landmark prediction model from Dlib.

Download the following model:

`shape_predictor_68_face_landmarks.dat`

You can obtain it from the official Dlib model repository:

[Download the 68-Face Landmark Model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

After downloading:

1. Extract the `.bz2` archive.
2. Place the extracted `.dat` file in the project directory.
3. Ensure that the model path in your Python script matches the file location.

Example:

```python
predictor_path = "shape_predictor_68_face_landmarks.dat"
```

### 5. Run the Application

Run the main Python script:

```bash
python main.py
```

> Replace `main.py` with the actual entry-point filename of your project.

The application should initialize your webcam and begin monitoring the driver's facial movements in real time.

## 📂 Project Structure

```text
AI-Powered-Driver-Monitoring-System/
│
├── assets/
│   └── demo.gif
│
├── main.py
├── shape_predictor_68_face_landmarks.dat
├── requirements.txt
└── README.md
```

> Update the project structure according to your actual files and directories.

## 🧠 Working Principle

The system uses computer vision techniques to monitor the driver's facial behavior.

### 1. Face Detection
The webcam captures real-time video frames. OpenCV processes the frames, while Dlib identifies the driver's face.

### 2. Facial Landmark Detection
Dlib's 68-point facial landmark predictor identifies key facial features, including the eyes, nose, and mouth.

### 3. Drowsiness Detection Using EAR
The Eye Aspect Ratio (EAR) is calculated using the vertical and horizontal distances between specific eye landmarks.

When the EAR falls below a predefined threshold for a certain duration, the system identifies potential drowsiness.

### 4. Head Pose Estimation
The system analyzes facial landmarks to estimate the driver's head orientation.

The estimated head pose helps determine whether the driver is looking forward or away from the road.

### 5. Alert Generation
When drowsiness or distraction is detected, the system triggers visual warnings and audible alerts to notify the driver.

## 🚀 Future Improvements

Potential enhancements for the project include:

- Integration with infrared cameras for low-light monitoring.
- Improved detection accuracy using deep learning models.
- Yawning detection using mouth aspect ratio (MAR).
- Detection of mobile phone usage while driving.
- Driver identity recognition.
- Integration with vehicle systems for advanced safety features.
- Performance optimization for embedded systems such as Raspberry Pi.

## ⚠️ Limitations

- Detection accuracy depends on lighting conditions and camera positioning.
- Facial occlusions, such as sunglasses or masks, may affect performance.
- Head pose estimation may not accurately represent the driver's visual attention.
- The system is intended as a prototype and should not be considered a replacement for certified automotive safety systems.

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

If you would like to contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Submit a pull request.

## 📜 License

This project is intended for educational and research purposes.

Add an appropriate open-source license, such as the MIT License, if you wish to allow others to use, modify, and distribute the project.

## 👨‍💻 Authors

Developed as a project focused on applying computer vision and artificial intelligence techniques to improve road safety.

---

**⭐ If you find this project interesting, consider giving the repository a star!**
