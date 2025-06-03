# HCI-Driven Learning: Hand Gesture Recognition for Learning Enhancement

An AI-powered educational game designed to make early learning more interactive, engaging, and accessible using hand and head gestures. Built for children aged 4–7, this project integrates computer vision and machine learning to create a kinesthetic learning environment using nothing more than a webcam and a standard laptop.

## Project Overview

This project combines deep learning, computer vision, and GUI development to deliver a three-phase educational platform that supports number recognition, vocabulary building, and logical thinking through gesture-based gameplay.

### Game Modules

1. **Finger Counting Game (CNN-Based)**
   - Children answer math/GK questions by holding up 0–5 fingers.
   - Real-time gesture recognition via a custom-trained CNN model.
   - Achieves ~94% training accuracy and ~89% validation accuracy.

2. **AI-Powered Crossword (Hand Tracking)**
   - Uses MediaPipe to track finger movement.
   - Children point and select letters on a virtual keyboard to solve puzzles.

3. **Synonym & Antonym Game (Head Tracking)**
   - Head tilts and blink detection used to select correct vocabulary options.
   - Completely hands-free interaction, making it highly inclusive.

## Technologies Used

- **Languages**: Python 3.8+
- **Libraries**: TensorFlow, OpenCV, MediaPipe, PyQt5, NumPy
- **Tools**: Jupyter Notebook, VS Code
- **Frameworks**: PyQt5 for GUI, custom CNN for gesture recognition

## How to Run

1. Clone the repo:
   git clone https://github.com/your-username/hci-driven-learning.git
   cd hci-driven-learning
   
2. pip install -r requirements.txt

3. python uimain.py
