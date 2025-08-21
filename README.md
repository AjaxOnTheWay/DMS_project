AI-Powered Driver Monitoring System
This is a real-time Driver Monitoring System (DMS) built with Python, OpenCV, and Dlib. It uses computer vision to detect signs of driver drowsiness and distraction, aiming to enhance road safety by providing timely visual and audible alerts.
Key Features
👁️ Real-Time Drowsiness Detection: Monitors the driver's Eye Aspect Ratio (EAR) to detect prolonged eye closure, a key indicator of microsleep.
🧠 Real-Time Distraction Detection: Utilizes Head Pose Estimation to determine if the driver is looking away from the road (left, right, or down).
🔊 Audible & Visual Alerts: Provides immediate visual warnings on-screen and plays a looping, stoppable alarm to effectively alert the driver.
🖥️ Real-Time Video Feed: Displays the live webcam feed with annotations for facial landmarks, eye contours, and head direction for clear visual feedback.
Demo
(A short GIF of the application running. This is highly recommended for your GitHub profile!)
![alt text](demo.gif)
Tech Stack
Python 3.x
OpenCV: For real-time computer vision, video capture, and drawing on the frame.
Dlib: For high-performance face detection and facial landmark prediction.
NumPy: For numerical operations, especially vector and matrix calculations for head pose estimation.
SciPy: For calculating the euclidean distance between facial landmarks.
Pygame: For handling the stoppable audible alerts.
Setup and Installation
Follow these steps to get the project running on your local machine.
1. Prerequisites
This project relies on libraries that need to be compiled from source. You must install the necessary build tools first.
CMake:
Download and install from the official CMake website.
Crucially, during installation, make sure to select the option "Add CMake to the system PATH for all users".
C++ Compiler:
Download the "Build Tools for Visual Studio" from the Visual Studio website.
Run the installer and select the "Desktop development with C++" workload.
2. Clone the Repository
code
Bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
3. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.
code
Bash
# Create the virtual environment
python -m venv dms_env

# Activate it (on Windows)
dms_env\Scripts\activate
4. Install Dependencies
Install all the required Python libraries using the provided requirements.txt file.
code
Bash
pip install -r requirements.txt
(Note: If you don't have a requirements.txt file, create one and add the following lines to it:)
code
Code
opencv-python
dlib
numpy
scipy
pygame
5. Download the Dlib Shape Predictor Model
The facial landmark detector requires a pre-trained model.
Download the model here: shape_predictor_68_face_landmarks.dat.bz2
Extract the shape_predictor_68_face_landmarks.dat file and place it in the root directory of the project.
6. Add an Alert Sound
Find an alert sound file (e.g., in .mp3 or .wav format).
Place it in the root directory of the project and name it alert.mp3, or change the ALERT_SOUND_PATH variable in dms.py to match your filename.
Usage
Once the setup is complete, run the main script from your activated virtual environment:
code
Bash
python dms.py
A window will appear showing your webcam feed.
To quit the application, press the 'q' key.
How It Works
Drowsiness Detection
The system tracks 68 facial landmarks in real-time. The Eye Aspect Ratio (EAR) is calculated using the vertical and horizontal distances of the eye landmarks. A sudden drop in the EAR value signifies a blink. If the EAR remains below a certain threshold for a set number of consecutive frames, the system triggers a drowsiness alert.
Distraction Detection
The system uses a 3D generic face model and the corresponding 2D facial landmarks from the video feed. By using OpenCV's solvePnP function, it calculates the head's rotation and translation vectors. This data is used to estimate the head's orientation and determine if the driver is looking forward, left, right, or down.
Future Improvements
Distraction Alert: Add a timer-based alert for when the driver looks away from the road for too long.
Yawn Detection: Implement a classifier to detect yawns as an additional indicator of fatigue.
Phone Detection: Train a simple object detection model to identify if the driver is holding a phone.
Web Dashboard: Integrate a Django or Flask backend to create a web-based dashboard for displaying driver statistics and alert logs.
Improve Robustness: Enhance performance in various lighting conditions and with drivers wearing glasses.
