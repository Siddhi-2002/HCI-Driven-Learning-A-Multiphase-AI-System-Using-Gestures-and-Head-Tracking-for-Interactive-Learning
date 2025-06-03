import cv2
import numpy as np
import math
import random
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model # Make sure this path is correct or handled
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QGraphicsDropShadowEffect, QVBoxLayout, QHBoxLayout, QApplication # Added QHBoxLayout, QApplication
from PyQt5.QtCore import QTimer, QPropertyAnimation, QRect, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QFont # Added QFont
import os
import sys
# Background (Global for this module or load as an asset)
underwater_bgr = np.full((600, 800, 3), (244, 194, 194), dtype=np.uint8)  # Baby pink BGR

# Combined Questions (can be part of the class or global to this module)
QUESTIONS = [
("How many eyes do you have?", 2),
("How many fingers do you have on one hand?", 5),
("How many legs do you have?", 2),
("How many thumbs do you have?", 2),
("How many noses do you have?", 1),
("How many mouths do you have?", 1),
("How many ears do you have?", 2),
("How many arms do you have?", 2),
("How many heads do you have?", 1),
("How many chins do you have?", 1),
("How many elbows do you have?", 2),
("How many hands do you have?", 2),
("How many feet do you have?", 2),
("How many knees do you have?", 2),
("How many hearts do you have?", 1),
("How many belly buttons do you have?", 1),
("How many pinky fingers do you have?", 2),
("How many nostrils do you have?", 2),
("How many toes do you have on one foot?", 5),
("How many eyebrows do you have?", 2),
("How many heads does a regular person have?", 1),
("How many fingers are NOT on one hand if it's in a glove with 2 missing?", 3),
("How many leaves are there on a cloverleaf?", 3),
("How many wheels does a tricycle have?", 3),
("How many legs does a chair usually have?", 4),
("How many paws does a cat have?", 4),
("How many corners does a square have?", 4),
("How many seasons are there in a year?", 4),
("What is 2 - 1?", 1),
("What is 2 + 1?", 3),
("What is 1 + 3?", 4),
("What is 2 + 2?", 4),
("What is 5 - 0?", 5),
("What is 4 + 1?", 5),
("What is 4 - 2?", 2),
("What is 0 + 4?", 4),
("What is 3 + 2?", 5),
("What is 1 + 2?", 3),
("What is 5 - 3?", 2),
("What is 0 + 5?", 5),
("What is 2 - 0?", 2),
("What is 4 - 1?", 3),
("What is 1 + 1?", 2),
("What is 3 - 1?", 2)
] 

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError: # Changed from Exception for more specificity
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def add_shadow(widget: QWidget, blur_radius=12, x_offset=4, y_offset=4, color=QColor(0, 0, 0, 160)):
    # ... (keep your add_shadow function)
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur_radius)
    shadow.setXOffset(x_offset)
    shadow.setYOffset(y_offset)
    shadow.setColor(color)
    widget.setGraphicsEffect(shadow)


class SpeechThread(QThread):
    # ... (keep your SpeechThread class)
    # Consider adding a parent to __init__ if it helps with thread management in complex apps
    speech_completed_signal = pyqtSignal() # Renamed for consistency with previous suggestions

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text_to_speak = text

    def run(self):
        try:
            engine = pyttsx3.init()
            engine.say(self.text_to_speak)
            engine.runAndWait()
        except Exception as e:
            print(f"SpeechThread Error: {e}")
        finally:
            self.speech_completed_signal.emit()


class ConfettiParticle(QLabel):
    # ... (keep your ConfettiParticle class)
    def __init__(self, parent):
        super().__init__(parent)
        self.color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.x = random.randint(0, parent.width())
        self.y = random.randint(-50, 0) # Start above the parent
        self.size = random.randint(8, 18) # Slightly larger confetti
        self.velocity = random.uniform(3, 7) # Slightly faster
        self.setFixedSize(self.size, self.size)
        self.move(int(self.x), int(self.y))
        self.show()

    def move_down(self):
        self.y += self.velocity
        if self.y > self.parent().height():
            # Reset to reuse particle
            self.y = random.randint(-50, 0)
            self.x = random.randint(0, self.parent().width())
            # self.hide() # Don't hide if reusing
        self.move(int(self.x), int(self.y))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self.size, self.size)


class HandGestureGame(QWidget): # Renamed from FingerCountingApp
    # Signal to tell MainApp to go back to its main menu
    back_to_main_menu_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.questions_all = QUESTIONS # Store the master list
        self.questions_current_round = [] # Questions for the current game instance

        self.question_number = 0
        self.current_fingers_count = 0
        self.answer_shown_this_attempt = False
        self.score = 0 # Add a score variable if you want to track it within the game

        self.model = None # Load in start_game
        self.detector = None # Init in start_game
        self.imgSize = 300
        self.offset = 20
        self.IMG_MODEL_SIZE = 128
        self.MODEL_PATH = r"C:\Users\Siddhi Mohanty\Desktop\Major Project\dependencies\model_30.keras" #resource_path(os.path.join("dependencies", "model_30.keras"))  Keep your path

        self.cap = None
        self.game_timer = QTimer(self) # Renamed from self.timer to avoid conflict
        self.game_timer.timeout.connect(self.update_frame)
        self.game_running = False

        # Background
        qimg_bg = QImage(underwater_bgr.data, underwater_bgr.shape[1], underwater_bgr.shape[0],
                         underwater_bgr.shape[1]*3, QImage.Format_BGR888)
        self.background_pixmap = QPixmap.fromImage(qimg_bg)

        self._initUI_elements() # Call the UI element creation method


    def _initUI_elements(self):
        """Initializes the UI elements and layouts."""
        # Camera Label
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(640, 360) # Smaller camera view to fit better
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("border: 3px solid #2e3b4e; background-color: black; border-radius: 10px;")

        # Question Label
        self.question_label = QLabel("Loading Question...", self)
        self.question_label.setAlignment(Qt.AlignCenter)
        self.question_label.setWordWrap(True)
        self.question_label.setStyleSheet("font-size: 24px; font-family: 'Comic Sans MS', Arial; color: #FFFFFF; background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5D3FD3, stop:1 #7B68EE); padding: 15px; border-radius: 15px; margin-bottom: 5px;")
        add_shadow(self.question_label, blur_radius=10, x_offset=3, y_offset=3, color=QColor(0,0,0,100))


        # Answer Label
        self.answer_label = QLabel("Show your answer with your fingers!", self)
        self.answer_label.setAlignment(Qt.AlignCenter)
        self.answer_label.setWordWrap(True)
        self.answer_label.setStyleSheet("font-size: 18px; font-family: Arial; color: #2e3b4e; background-color: rgba(255, 255, 255, 0.85); padding: 10px; border-radius: 10px; margin-top: 5px;")
        add_shadow(self.answer_label, blur_radius=8, x_offset=2, y_offset=2, color=QColor(0,0,0,80))

        # Score Label
        self.score_label_display = QLabel(f"Score: {self.score}", self)
        self.score_label_display.setAlignment(Qt.AlignCenter)
        self.score_label_display.setStyleSheet("font-size: 20px; font-family: Arial; color: #FFFFFF; background-color: #2e3b4e; padding: 8px; border-radius: 8px; margin-top:10px;")
        add_shadow(self.score_label_display)


        # Next Button
        self.next_button = QPushButton("Next Question", self)
        self.next_button.setMinimumHeight(55)
        self.next_button.setStyleSheet("""QPushButton {background-color: #66BB6A; color: white; font-size: 18px; font-weight: bold; border-radius: 12px; padding: 10px;} QPushButton:hover {background-color: #4CAF50;} QPushButton:pressed {background-color: #388E3C;}""")
        self.next_button.clicked.connect(self.next_question_logic)
        add_shadow(self.next_button)

        # Reward Button
        self.reward_button = QPushButton("🎁 Show My Reward! 🎁", self)
        self.reward_button.setMinimumHeight(55)
        self.reward_button.setStyleSheet("""QPushButton {background-color: #BA68C8; color: white; font-size: 18px; font-weight: bold; border-radius: 12px; padding: 10px;} QPushButton:hover {background-color: #9C27B0;} QPushButton:pressed {background-color: #7B1FA2;}""")
        self.reward_button.setVisible(False)
        self.reward_button.clicked.connect(self.trigger_reward_animation)
        add_shadow(self.reward_button)

        # Back to Menu Button (emits signal)
        self.back_button = QPushButton("<< Back to Menu", self)
        self.back_button.setMinimumHeight(45)
        # Use parent's BACK_BTN_STYLE if available, else fallback to a default
        style = getattr(self.parent(), "BACK_BTN_STYLE", """
QPushButton {
    font-size: 20px;
    font-weight: bold;
    color: white;
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8E44AD, stop:1 #5E3370);
    border: 2px solid #5E3370;
    border-radius: 12px;
    padding: 18px 36px;
    min-width: 220px;
}
QPushButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #A569BD, stop:1 #7D3C98);
}
QPushButton:pressed {
    background-color: #5E3370;
}
""")
        self.back_button.setStyleSheet(style)
        self.back_button.clicked.connect(self.go_back_to_main_menu)
        add_shadow(self.back_button)

        # Layout Setup
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20) # Add some overall padding
        main_layout.setSpacing(15)

        main_layout.addWidget(self.question_label, 0, Qt.AlignTop)
        main_layout.addWidget(self.answer_label, 0, Qt.AlignTop)
        main_layout.addWidget(self.score_label_display,0, Qt.AlignTop | Qt.AlignHCenter)
        main_layout.addStretch(1) # Pushes camera view down a bit
        main_layout.addWidget(self.camera_label, 0, Qt.AlignCenter)
        main_layout.addStretch(1) # Spacer

        button_hbox = QHBoxLayout()
        button_hbox.addWidget(self.back_button)
        button_hbox.addStretch(1)
        button_hbox.addWidget(self.reward_button) # Reward button can share space with Next
        button_hbox.addWidget(self.next_button)
        main_layout.addLayout(button_hbox)

        self.setLayout(main_layout)

        # Confetti setup
        self.confetti_particles = []
        self.confetti_timer = QTimer(self)
        self.confetti_timer.timeout.connect(self.update_confetti_animation)


    def start_game_or_reset(self): # Keep this name as ui_main.py expects it
        print("HandGestureGame: Starting/Resetting.")
        self.questions_current_round = random.sample(self.questions_all, min(10, len(self.questions_all)))
        if not self.questions_current_round:
            self.question_label.setText("Error: No questions loaded!")
            return

        self.question_number = 0
        self.score = 0
        self.update_score_display()
        self.answer_shown_this_attempt = False
        self.reward_button.setVisible(False)
        self.next_button.setText("Next Question")
        self.next_button.setEnabled(True) # Make sure it's clickable

        if self.model is None:
            try:
                self.model = load_model(self.MODEL_PATH)
                print(f"Model loaded from {self.MODEL_PATH}")
            except Exception as e:
                print(f"FATAL: Could not load Keras model from {self.MODEL_PATH} - {e}")
                self.question_label.setText("Error: Model Load Failed!")
                self.answer_label.setText(f"Path: {self.MODEL_PATH}")
                return
        if self.detector is None:
            self.detector = HandDetector(maxHands=1, detectionCon=0.8) # Use desired confidence

        if self.cap is None:
            self.cap = cv2.VideoCapture(0) # Default camera
            if not self.cap.isOpened():
                print("Error: Cannot open camera for HandGestureGame.")
                self.cap = None
                self.camera_label.setText("Camera Error")
                return
        self.camera_label.setText("Loading Camera...") # Placeholder text
        self.game_timer.start(33) # ~30 FPS
        self.game_running = True
        self.display_current_question()
        self.setFocus()

    def stop_game_or_pause(self):
        print("HandGestureGame: Stopping/Pausing.")
        self.game_running = False
        if self.game_timer.isActive():
            self.game_timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'speech_thread') and self.speech_thread.isRunning():
            self.speech_thread.quit()
            self.speech_thread.wait()
        self.stop_confetti()

    def display_current_question(self):
        if self.question_number < len(self.questions_current_round):
            q_text = self.questions_current_round[self.question_number][0]
            self.question_label.setText(q_text)
            self.answer_label.setText("Show your answer with your fingers!")
            self.answer_shown_this_attempt = False
            self.speak_question(q_text)
        else:
            self.question_label.setText("🎉 You've completed all questions! 🎉")
            self.answer_label.setText("Click the reward button!")
            self.reward_button.setVisible(True)
            self.next_button.setEnabled(False)
            # self.stop_game_or_pause() # Don't stop game yet, allow reward

    def speak_question(self, question_text):
        # ... (keep your speak_question logic)
        if hasattr(self, 'speech_thread') and self.speech_thread.isRunning():
            self.speech_thread.terminate() # Force stop if still running
            self.speech_thread.wait()
        self.speech_thread = SpeechThread(question_text, self)
        self.speech_thread.speech_completed_signal.connect(self.on_speech_completed)
        self.speech_thread.start()


    def on_speech_completed(self):
        # print("Speech completed.") # Debug
        pass

    def update_frame(self):
        # ... (your existing update_frame logic for hand detection and prediction)
        # Ensure you use self.questions_current_round
        # And update self.score and call self.update_score_display()
        if not self.game_running or self.cap is None or not self.cap.isOpened():
            return
        if self.question_number >= len(self.questions_current_round) and not self.reward_button.isVisible(): # Game logic part ended
             self.display_current_question() # This will show the "completed" message
             return


        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        hands, img_for_detector = self.detector.findHands(frame, flipType=False)
        display_frame = img_for_detector.copy()  # Use the image with landmarks drawn

        current_detected_fingers = 0

        if hands:
            main_hand = hands[0]
            x, y, w, h = main_hand['bbox']
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            y_start, y_end = max(0, y - self.offset), min(frame.shape[0], y + h + self.offset)
            x_start, x_end = max(0, x - self.offset), min(frame.shape[1], x + w + self.offset)
            imgCrop = frame[y_start:y_end, x_start:x_end]

            if imgCrop.size > 0:
                aspectRatio = imgCrop.shape[0] / imgCrop.shape[1] if imgCrop.shape[1] > 0 else 0
                try:
                    if aspectRatio > 1:
                        k = self.imgSize / imgCrop.shape[0] if imgCrop.shape[0] > 0 else 0
                        wCal = math.ceil(k * imgCrop.shape[1])
                        if wCal > 0:
                            imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                            wGap = (self.imgSize - wCal) // 2
                            imgWhite[:, max(0,wGap):min(self.imgSize, wCal + wGap)] = imgResize[:, :min(wCal, imgWhite.shape[1]-max(0,wGap))]
                    else:
                        k = self.imgSize / imgCrop.shape[1] if imgCrop.shape[1] > 0 else 0
                        hCal = math.ceil(k * imgCrop.shape[0])
                        if hCal > 0:
                            imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                            hGap = (self.imgSize - hCal) // 2
                            imgWhite[max(0,hGap):min(self.imgSize, hCal + hGap), :] = imgResize[:min(hCal, imgWhite.shape[0]-max(0,hGap)),:]

                    if self.model:
                        imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
                        imgInput_prep = cv2.resize(imgGray, (self.IMG_MODEL_SIZE, self.IMG_MODEL_SIZE))
                        imgInput_prep = imgInput_prep / 255.0
                        imgInput_final = np.expand_dims(imgInput_prep, axis=(0, -1))
                        prediction = self.model.predict(imgInput_final)
                        current_detected_fingers = np.argmax(prediction)
                except Exception as e:
                    print(f"Hand Processing/Prediction Error: {e}")
                    current_detected_fingers = -1 # Error state
            else: # imgCrop was empty
                current_detected_fingers = 0 # or -1 to indicate no hand clearly visible
        # Update self.current_fingers_count only if a new valid detection happened
        if hands or current_detected_fingers == -1 : # if hand detected or error happened
             self.current_fingers_count = current_detected_fingers


        # Update answer label (only if not already answered correctly for this attempt)
        if self.question_number < len(self.questions_current_round):
            correct_answer = self.questions_current_round[self.question_number][1]
            if self.current_fingers_count > 0:
                if self.current_fingers_count == correct_answer and not self.answer_shown_this_attempt:
                    self.answer_label.setText(f"✅ Correct! You showed {self.current_fingers_count} fingers!")
                    self.answer_shown_this_attempt = True
                    self.score += 10
                    self.update_score_display()
                elif not self.answer_shown_this_attempt:
                    self.answer_label.setText("❌ Try again! Show the correct number of fingers.")
            elif self.current_fingers_count == -1 and not self.answer_shown_this_attempt:
                self.answer_label.setText("Error processing. Try again.")
            elif not self.answer_shown_this_attempt:
                self.answer_label.setText("Show your answer with your fingers!")


        # Display frame
        qimg = QImage(display_frame.data, display_frame.shape[1], display_frame.shape[0],
                      display_frame.strides[0], QImage.Format_BGR888)
        self.camera_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def next_question_logic(self):
        # if not self.answer_shown_this_attempt and self.question_number < len(self.questions_current_round):
            # Logic for what happens if "Next" is pressed before correct answer shown.
            # Currently, it will just advance. Consider if you want to penalize or force correct answer.
            # print("Next pressed, answer shown:", self.answer_shown_this_attempt)

        self.question_number += 1
        self.answer_shown_this_attempt = False # Reset for the new question
        self.display_current_question() # This handles UI update for new question or end game

    def trigger_reward_animation(self):
        # ... (keep your reward logic, maybe adjust confetti parent)
        self.answer_label.setText("⭐ Amazing job! ⭐")
        animation = QPropertyAnimation(self.reward_button, b"geometry", self)
        animation.setDuration(1000)
        animation.setStartValue(self.reward_button.geometry())
        animation.setEndValue(QRect(self.reward_button.x() - 10, self.reward_button.y() - 10,
                                    self.reward_button.width() + 20, self.reward_button.height() + 20))
        animation.start()
        self.start_confetti()


    def start_confetti(self):
        # ... (keep your confetti start logic, make sure confetti parent is `self`)
        if not self.confetti_timer.isActive():
            for p in self.confetti_particles: p.deleteLater() # Clear old ones
            self.confetti_particles.clear()
            for _ in range(100):
                particle = ConfettiParticle(self) # Parent is now the HandGestureGame widget
                self.confetti_particles.append(particle)
            self.confetti_timer.start(30)

    def stop_confetti(self):
        # ... (keep your confetti stop logic)
        if self.confetti_timer.isActive():
            self.confetti_timer.stop()
        for p in self.confetti_particles:
            p.hide()


    def update_confetti_animation(self):
        # ... (keep your confetti update logic)
        all_hidden = True
        for particle in self.confetti_particles:
            if particle.isVisible(): # Check if still visible before moving
                particle.move_down()
                if particle.isVisible(): # Check again after move_down
                    all_hidden = False
        if all_hidden and self.confetti_particles:
            self.stop_confetti()

    def go_back_to_main_menu(self): # New handler for back button
        self.stop_game_or_pause() # Ensure game stops cleanly
        self.back_to_main_menu_signal.emit() # Emit signal for MainApp

    def update_score_display(self):
        self.score_label_display.setText(f"Score: {self.score}")

    # Override paintEvent to draw the background for the QWidget
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.background_pixmap)
        # No need to call super().paintEvent(event) if all children are managed by layouts
        # and you don't have custom QWidget children painting themselves outside layouts.

    def resizeEvent(self, event):
        # This is where you would reposition/resize UI elements if NOT using layouts primarily
        # Or if your background needs to be rescaled.
        # For now, the layout should handle most of it.
        # Ensure the background scales if the widget itself can resize.
        # self.background_label.setPixmap(self.background_pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        super().resizeEvent(event)

    # For ui_main.py to interact with specific keys if needed
    def customKeyPressEvent(self, event):
        key = event.key()
        if self.game_running and not self.reward_button.isVisible(): # Only if game is active and not at final reward
            if key == Qt.Key_N: # 'N' for next question via keyboard
                self.next_question_logic()
                return True # Event was handled
        # Could add 'R' to reset game within this screen if desired
        # if key == Qt.Key_R:
        #     self.start_game_or_reset(question_set=self.active_question_set, main_app_callback=self.main_app_callback)
        #     return True
        return False # Event not handled here

    def closeEvent(self, event): # QWidget uses closeEvent not just QMainWindow
        self.stop_game_or_pause()
        super().closeEvent(event)

    def hideEvent(self, event): # Also good to stop when hidden by stacked widget
        self.stop_game_or_pause()
        super().hideEvent(event)



# This standalone test block will be removed when integrating into ui_main.py
# but is useful for testing hand_gesture_game.py on its own.
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    # Dummy callback for testing
    def dummy_main_menu_switch():
        print("Callback: Would switch to main menu now.")
        # In a real test, you might close the game_widget
        # game_widget.close() # This would trigger its closeEvent

    game_widget = HandGestureGame()
    game_widget.back_to_main_menu_signal.connect(dummy_main_menu_switch) # Connect signal for test
    game_widget.start_game_or_reset() # Default to STAGE1, sets dummy callback

    game_widget.setWindowTitle("Test Hand Gesture Game Standalone")
    game_widget.resize(800, 700) # Give a bit more height for all elements
    game_widget.show()
    sys.exit(app.exec_())