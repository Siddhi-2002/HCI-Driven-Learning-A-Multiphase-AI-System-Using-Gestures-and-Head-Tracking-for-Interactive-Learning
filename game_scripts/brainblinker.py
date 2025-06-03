# brain_blinker_game.py

import cv2
import mediapipe as mp
import numpy as np
import time
import random
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer, QPoint , QSize # QPoint might not be needed here directly
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal

class BrainBlinkerGame(QWidget):
    back_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Brain Blinker: Synonym/Antonym Hunter") # Or keep it generic for your "Tilt & Win" quiz

        # --- OpenCV/MediaPipe and Game Variables (now instance variables) ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None # Initialize in start_game
        self.cap = None       # Initialize in start_game

        # Eye Blink Detection
        self.EYE_AR_THRESH = 0.20
        self.EYE_AR_CONSEC_FRAMES = 2
        self.blink_counter = 0
        # self.eyes_closed = False # Not strictly needed with current raw_blink logic
        self.raw_blink_detected_this_frame = False
        self.LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_EAR_INDICES = [33, 158, 160, 133, 144, 153]
        self.NOSE_TIP_INDEX = 4

        # Head Control
        self.initial_nose_x = None
        self.initial_nose_y = None
        self.HEAD_SENSITIVITY = 5.0
        self.SMOOTHING_FACTOR = 0.6 # Adjusted from previous examples for more stability
        self.FRAME_WIDTH_CAM = 800 # Match your OpenCV game
        self.FRAME_HEIGHT_CAM = 600
        self.pointer_x = self.FRAME_WIDTH_CAM // 2
        self.pointer_y = self.FRAME_HEIGHT_CAM // 2
        self.pointer_radius = 10


        # Dwell + Blink
        self.DWELL_TO_ARM_TIME = 0.4
        self.armed_word_idx = -1
        self.dwell_start_time = 0
        self.current_highlighted_for_dwell_idx = -1

        # Game Data
        self.WORD_DATA = {
            "happy": {"synonyms": ["joyful", "glad", "pleased", "content", "elated"], "antonyms": ["sad", "unhappy", "miserable", "dejected"]},
            "big": {"synonyms": ["large", "huge", "enormous", "giant", "massive"], "antonyms": ["small", "tiny", "little", "miniature"]},
            # ... (add the rest of your WORD_DATA)
             "fast": {"synonyms": ["quick", "swift", "rapid", "speedy", "brisk"], "antonyms": ["slow", "sluggish", "leisurely"]},
            "good": {"synonyms": ["fine", "nice", "great", "excellent", "superb"], "antonyms": ["bad", "poor", "terrible", "awful"]},
            "brave": {"synonyms": ["courageous", "bold", "fearless", "valiant"], "antonyms": ["cowardly", "timid", "fearful"]},
            "bright": {"synonyms": ["shiny", "radiant", "luminous", "vivid"], "antonyms": ["dark", "dim", "dull"]},
            "important": {"synonyms": ["significant", "crucial", "vital", "essential"], "antonyms": ["trivial", "minor", "unimportant"]},
            "difficult": {"synonyms": ["hard", "challenging", "tough", "arduous"], "antonyms": ["easy", "simple", "effortless"]},
        }
        self.ALL_WORDS = list(self.WORD_DATA.keys())
        for data in self.WORD_DATA.values():
            self.ALL_WORDS.extend(data['synonyms'])
            self.ALL_WORDS.extend(data['antonyms'])
        self.ALL_WORDS = list(set(self.ALL_WORDS))

        # Game State
        self.current_target_word = ""
        self.current_mode = "synonyms"
        self.word_grid = []
        self.score = 0
        self.round_timer_start = 0
        self.ROUND_DURATION = 35
        self.feedback_message = ""
        self.feedback_color = (255, 255, 255)
        self.feedback_end_time = 0
        self.game_over = False
        self.game_running = False # To control the QTimer

        # UI Constants for OpenCV drawing
        self.GRID_COLS = 4
        self.GRID_ROWS = 3
        self.CELL_PADDING = 20
        self.HEADER_HEIGHT = 100
        self.GRID_START_Y = self.HEADER_HEIGHT
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE_WORD = 0.65
        self.FONT_THICKNESS = 2
        # --- End of Game Variables ---

        # --- PyQt5 UI Setup ---
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0) ### MODIFIED ###
        self.layout.setSpacing(0) # The QLabel will take the full space
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.video_label, stretch=1)

        # Back to Menu button
        self.back_button = QPushButton("<< Back to Menu", self)
        self.back_button.setStyleSheet("""
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
        self.back_button.clicked.connect(self.go_to_menu)
        self.layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

        # Next button
        self.next_button = QPushButton("Next", self)
        self.next_button.setStyleSheet("""
    QPushButton {
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #4CAF50;
        border-radius: 10px;
        padding: 10px 30px;
    }
    QPushButton:hover { background-color: #388E3C; }
""")
        self.next_button.clicked.connect(self.next_round)
        self.layout.addWidget(self.next_button, alignment=Qt.AlignCenter)

        # Game loop timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        # --- End of PyQt5 UI Setup ---

        # self.setup_new_round() # Call this in start_game_or_reset

        # Additional variables for tracking selections
        self.correct_selections = 0
        self.total_selections = 0

        # Error count for tracking issues
        self.error_count = 0

        # Frame timing for performance analysis
        self.frame_times = []

        # Blink detection metrics
        self.blinks_detected = 0
        self.frames_processed = 0

        # Face detection frame count
        self.face_detected_frames = 0

        self.WORD_DATA_ORIGINAL = {k: v.copy() for k, v in self.WORD_DATA.items()}

    def calculate_ear(self, eye_landmarks_normalized, frame_shape):
        # (Same as your original function)
        h, w = frame_shape[:2]
        coords = np.array([(lm.x * w, lm.y * h) for lm in eye_landmarks_normalized])
        if len(coords) < 6: return 1.0
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h1 = np.linalg.norm(coords[0] - coords[3])
        if h1 == 0: return 1.0
        ear = (v1 + v2) / (2.0 * h1)
        return ear

    def display_feedback(self, message, color, duration=2):
        self.feedback_message = message
        self.feedback_color = color
        self.feedback_end_time = time.time() + duration

    def get_distractor_words(self, correct_words, num_distractors=8):
        # (Same as your original function, but use self.ALL_WORDS, self.current_target_word)
        distractors = []
        possible_distractors = [w for w in self.ALL_WORDS if w not in correct_words and w != self.current_target_word]
        random.shuffle(possible_distractors)
        return possible_distractors[:num_distractors]


    def setup_new_round(self):
        # (Same as your original function, but use self. for all game state vars)
        # e.g., self.word_grid.clear(), self.current_target_word = ..., etc.
        if not self.WORD_DATA:
            self.end_game("No more words!")
            return

        self.feedback_message = ""
        self.armed_word_idx = -1
        self.current_target_word = random.choice(list(self.WORD_DATA.keys()))
        target_data = self.WORD_DATA[self.current_target_word]

        if self.current_mode == "synonyms":
            correct_options = target_data['synonyms']
        else:
            correct_options = target_data['antonyms']

        if not correct_options:
            print(f"Warning: No {self.current_mode} for {self.current_target_word}. Skipping.")
            # Make sure WORD_DATA still has keys before trying to delete
            if self.current_target_word in self.WORD_DATA:
                del self.WORD_DATA[self.current_target_word]
            if self.WORD_DATA: self.setup_new_round()
            else: self.end_game("Ran out of suitable words!")
            return

        num_correct_to_show = min(len(correct_options), (self.GRID_COLS * self.GRID_ROWS // 2) +1 )
        selected_correct = random.sample(correct_options, num_correct_to_show)

        num_distractors_needed = (self.GRID_COLS * self.GRID_ROWS) - num_correct_to_show
        all_target_related_words = [self.current_target_word] + target_data['synonyms'] + target_data['antonyms']
        distractor_pool = [w for w in self.ALL_WORDS if w not in all_target_related_words]
        random.shuffle(distractor_pool)
        distractors = distractor_pool[:num_distractors_needed]

        options_for_grid = selected_correct + distractors
        random.shuffle(options_for_grid)

        self.word_grid.clear()
        cell_width = self.FRAME_WIDTH_CAM // self.GRID_COLS
        cell_height = (self.FRAME_HEIGHT_CAM - self.GRID_START_Y) // self.GRID_ROWS

        for i, word_text in enumerate(options_for_grid):
            if i >= self.GRID_COLS * self.GRID_ROWS: break
            row = i // self.GRID_COLS
            col = i % self.GRID_COLS
            cell_x_start = col * cell_width
            cell_y_start = self.GRID_START_Y + row * cell_height
            content_x = cell_x_start + self.CELL_PADDING
            content_y = cell_y_start + self.CELL_PADDING
            content_w = cell_width - 2 * self.CELL_PADDING
            content_h = cell_height - 2 * self.CELL_PADDING

            self.word_grid.append({
                'text': word_text, 'display_text': word_text,
                'x': content_x, 'y': content_y, 'w': content_w, 'h': content_h,
                'cell_x': cell_x_start, 'cell_y': cell_y_start, 'cell_w': cell_width, 'cell_h': cell_height,
                'is_correct': (word_text in selected_correct),
                'is_selected': False, 'is_revealed': False
            })
        self.round_timer_start = time.time()
        for item in self.word_grid: # Ellipsis logic
            available_width = item['w'] - 10
            original_text = item['text']
            font_scale = self.FONT_SCALE_WORD; thickness = self.FONT_THICKNESS
            text_size, _ = cv2.getTextSize(original_text, self.FONT, font_scale, thickness)
            if text_size[0] > available_width:
                temp_text = original_text
                while len(temp_text) > 0:
                    temp_text = temp_text[:-1]
                    current_display = temp_text + "...";
                    text_size, _ = cv2.getTextSize(current_display, self.FONT, font_scale, thickness)
                    if text_size[0] <= available_width:
                        item['display_text'] = current_display; break
                if len(temp_text) == 0 : item['display_text'] = "..."
            else: item['display_text'] = original_text

    def end_game(self, message="Game Over!"):

        max_score = self.get_max_score()
        print(f"Your Score: {self.score}")
        print(f"Max Possible Score: {max_score}")
        self.feedback_message = f"All done!\nYour Score: {self.score}\nMax Score: {max_score}"

        self.game_over = True
        self.next_button.setEnabled(False)
        # --- PyQt5 UI Setup ---
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0) ### MODIFIED ###
        self.layout.setSpacing(0) # The QLabel will take the full space
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.video_label, stretch=1)

        # Back to Menu button
        self.back_button = QPushButton("<< Back to Menu", self)
        self.back_button.setStyleSheet("""
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
        self.back_button.clicked.connect(self.go_to_menu)
        self.layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

        # Next button
        self.next_button = QPushButton("Next", self)
        self.next_button.setStyleSheet("""
    QPushButton {
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #4CAF50;
        border-radius: 10px;
        padding: 10px 30px;
    }
    QPushButton:hover { background-color: #388E3C; }
""")
        self.next_button.clicked.connect(self.next_round)
        self.layout.addWidget(self.next_button, alignment=Qt.AlignCenter)

        # Game loop timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        # --- End of PyQt5 UI Setup ---

        # self.setup_new_round() # Call this in start_game_or_reset

        # Additional variables for tracking selections
        self.correct_selections = 0
        self.total_selections = 0

        # Error count for tracking issues
        self.error_count = 0

        # Frame timing for performance analysis
        self.frame_times = []

        # Blink detection metrics
        self.blinks_detected = 0
        self.frames_processed = 0

        # Face detection frame count
        self.face_detected_frames = 0

        self.WORD_DATA_ORIGINAL = {k: v.copy() for k, v in self.WORD_DATA.items()}

    def calculate_ear(self, eye_landmarks_normalized, frame_shape):
        # (Same as your original function)
        h, w = frame_shape[:2]
        coords = np.array([(lm.x * w, lm.y * h) for lm in eye_landmarks_normalized])
        if len(coords) < 6: return 1.0
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h1 = np.linalg.norm(coords[0] - coords[3])
        if h1 == 0: return 1.0
        ear = (v1 + v2) / (2.0 * h1)
        return ear

    def display_feedback(self, message, color, duration=2):
        self.feedback_message = message
        self.feedback_color = color
        self.feedback_end_time = time.time() + duration

    def get_distractor_words(self, correct_words, num_distractors=8):
        # (Same as your original function, but use self.ALL_WORDS, self.current_target_word)
        distractors = []
        possible_distractors = [w for w in self.ALL_WORDS if w not in correct_words and w != self.current_target_word]
        random.shuffle(possible_distractors)
        return possible_distractors[:num_distractors]


    def setup_new_round(self):
        # (Same as your original function, but use self. for all game state vars)
        # e.g., self.word_grid.clear(), self.current_target_word = ..., etc.
        if not self.WORD_DATA:
            self.end_game("No more words!")
            return

        self.feedback_message = ""
        self.armed_word_idx = -1
        self.current_target_word = random.choice(list(self.WORD_DATA.keys()))
        target_data = self.WORD_DATA[self.current_target_word]

        if self.current_mode == "synonyms":
            correct_options = target_data['synonyms']
        else:
            correct_options = target_data['antonyms']

        if not correct_options:
            print(f"Warning: No {self.current_mode} for {self.current_target_word}. Skipping.")
            # Make sure WORD_DATA still has keys before trying to delete
            if self.current_target_word in self.WORD_DATA:
                del self.WORD_DATA[self.current_target_word]
            if self.WORD_DATA: self.setup_new_round()
            else: self.end_game("Ran out of suitable words!")
            return

        num_correct_to_show = min(len(correct_options), (self.GRID_COLS * self.GRID_ROWS // 2) +1 )
        selected_correct = random.sample(correct_options, num_correct_to_show)

        num_distractors_needed = (self.GRID_COLS * self.GRID_ROWS) - num_correct_to_show
        all_target_related_words = [self.current_target_word] + target_data['synonyms'] + target_data['antonyms']
        distractor_pool = [w for w in self.ALL_WORDS if w not in all_target_related_words]
        random.shuffle(distractor_pool)
        distractors = distractor_pool[:num_distractors_needed]

        options_for_grid = selected_correct + distractors
        random.shuffle(options_for_grid)

        self.word_grid.clear()
        cell_width = self.FRAME_WIDTH_CAM // self.GRID_COLS
        cell_height = (self.FRAME_HEIGHT_CAM - self.GRID_START_Y) // self.GRID_ROWS

        for i, word_text in enumerate(options_for_grid):
            if i >= self.GRID_COLS * self.GRID_ROWS: break
            row = i // self.GRID_COLS
            col = i % self.GRID_COLS
            cell_x_start = col * cell_width
            cell_y_start = self.GRID_START_Y + row * cell_height
            content_x = cell_x_start + self.CELL_PADDING
            content_y = cell_y_start + self.CELL_PADDING
            content_w = cell_width - 2 * self.CELL_PADDING
            content_h = cell_height - 2 * self.CELL_PADDING

            self.word_grid.append({
                'text': word_text, 'display_text': word_text,
                'x': content_x, 'y': content_y, 'w': content_w, 'h': content_h,
                'cell_x': cell_x_start, 'cell_y': cell_y_start, 'cell_w': cell_width, 'cell_h': cell_height,
                'is_correct': (word_text in selected_correct),
                'is_selected': False, 'is_revealed': False
            })
        self.round_timer_start = time.time()
        for item in self.word_grid: # Ellipsis logic
            available_width = item['w'] - 10
            original_text = item['text']
            font_scale = self.FONT_SCALE_WORD; thickness = self.FONT_THICKNESS
            text_size, _ = cv2.getTextSize(original_text, self.FONT, font_scale, thickness)
            if text_size[0] > available_width:
                temp_text = original_text
                while len(temp_text) > 0:
                    temp_text = temp_text[:-1]
                    current_display = temp_text + "...";
                    text_size, _ = cv2.getTextSize(current_display, self.FONT, font_scale, thickness)
                    if text_size[0] <= available_width:
                        item['display_text'] = current_display; break
                if len(temp_text) == 0 : item['display_text'] = "..."
            else: item['display_text'] = original_text

    def end_game(self, message="Game Over!"):

        self.game_over = True
        self.next_button.setEnabled(False)

    def update_frame(self):
        start_time = time.time()
        self.frames_processed += 1  # <-- Add this line!
        if not self.game_running or self.game_over:
            # Optionally draw a "Game Over" or "Paused" screen if timer is still running
            # For now, just return if not actively running the game logic
            if self.game_over:
                 # Create a blank frame or a specific game over image
                frame = np.full((self.FRAME_HEIGHT_CAM, self.FRAME_WIDTH_CAM, 3), (80, 50, 20), dtype=np.uint8)
                text_size_feedback, _ = cv2.getTextSize(self.feedback_message, self.FONT, 1.0, 2)
                msg_x = (self.FRAME_WIDTH_CAM - text_size_feedback[0]) // 2
                msg_y = self.FRAME_HEIGHT_CAM // 2
                cv2.putText(frame, self.feedback_message, (msg_x, msg_y), self.FONT, 1.0, self.feedback_color, 2)
                self.display_cv_frame(frame)
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error reading frame from camera.")
            # self.stop_game_or_pause() # Consider stopping if camera fails
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # For MediaPipe
        results = self.face_mesh.process(frame_rgb)
        self.raw_blink_detected_this_frame = False # Reset

        if results.multi_face_landmarks:
            self.face_detected_frames += 1
            face_landmarks = results.multi_face_landmarks[0].landmark
            nose_tip_lm = face_landmarks[self.NOSE_TIP_INDEX]
            nose_x_norm, nose_y_norm = nose_tip_lm.x, nose_tip_lm.y

            if self.initial_nose_x is None:
                self.initial_nose_x, self.initial_nose_y = nose_x_norm, nose_y_norm

            delta_x = nose_x_norm - self.initial_nose_x
            delta_y = nose_y_norm - self.initial_nose_y

            target_pointer_x = (self.FRAME_WIDTH_CAM / 2) + (delta_x * self.FRAME_WIDTH_CAM * self.HEAD_SENSITIVITY)
            target_pointer_y = (self.FRAME_HEIGHT_CAM / 2) + (delta_y * self.FRAME_HEIGHT_CAM * self.HEAD_SENSITIVITY)

            self.pointer_x = int((1 - self.SMOOTHING_FACTOR) * target_pointer_x + self.SMOOTHING_FACTOR * self.pointer_x)
            self.pointer_y = int((1 - self.SMOOTHING_FACTOR) * target_pointer_y + self.SMOOTHING_FACTOR * self.pointer_y)

            self.pointer_x = np.clip(self.pointer_x, 0, self.FRAME_WIDTH_CAM - 1)
            self.pointer_y = np.clip(self.pointer_y, 0, self.FRAME_HEIGHT_CAM - 1)

            try:
                left_ear = self.calculate_ear([face_landmarks[i] for i in self.LEFT_EYE_EAR_INDICES], frame.shape)
                right_ear = self.calculate_ear([face_landmarks[i] for i in self.RIGHT_EYE_EAR_INDICES], frame.shape)
                avg_ear = (left_ear + right_ear) / 2.0
                if avg_ear < self.EYE_AR_THRESH:
                    self.blink_counter += 1
                    if self.blink_counter >= self.EYE_AR_CONSEC_FRAMES:
                        self.raw_blink_detected_this_frame = True
                        self.blinks_detected += 1  # <-- Add this line!
                else:
                    self.blink_counter = 0
            except Exception as e:
                print(f"EAR calculation error: {e}") # For debugging
                pass

        # --- Game Logic (Copied from your loop, using self. for variables) ---
        time_elapsed_round = time.time() - self.round_timer_start
        time_remaining = max(0, self.ROUND_DURATION - int(time_elapsed_round))

        if time_remaining <= 0 and not self.game_over:
            self.display_feedback(f"Time's up for '{self.current_target_word}'!", (255,165,0))
            if self.current_target_word in self.WORD_DATA: del self.WORD_DATA[self.current_target_word]
            if not self.WORD_DATA: self.end_game("All words done!")
            else: self.setup_new_round()

        current_hovered_word_idx = -1
        for i, word_info in enumerate(self.word_grid):
            if not word_info['is_revealed'] and \
               self.pointer_x > word_info['x'] and self.pointer_x < word_info['x'] + word_info['w'] and \
               self.pointer_y > word_info['y'] and self.pointer_y < word_info['y'] + word_info['h']:
                current_hovered_word_idx = i; break

        if current_hovered_word_idx != -1:
            if self.current_highlighted_for_dwell_idx == current_hovered_word_idx:
                if self.armed_word_idx != current_hovered_word_idx and time.time() - self.dwell_start_time >= self.DWELL_TO_ARM_TIME:
                    self.armed_word_idx = current_hovered_word_idx
            else:
                self.current_highlighted_for_dwell_idx = current_hovered_word_idx
                self.dwell_start_time = time.time()
                self.armed_word_idx = -1
        else:
            self.current_highlighted_for_dwell_idx = -1
            self.armed_word_idx = -1
            self.dwell_start_time = time.time()

        actual_selection_this_frame = False # You had this, might be useful
        if self.raw_blink_detected_this_frame and self.armed_word_idx != -1 and not self.game_over:
            selected_word_info = self.word_grid[self.armed_word_idx]
            if not selected_word_info['is_revealed']:
                self.total_selections += 1
                if selected_word_info['is_correct']:
                    self.correct_selections += 1
                    self.score += 10
                    self.display_feedback("Correct!", (0, 200, 0))
                else:
                    self.score -= 5
                    self.display_feedback("Incorrect.", (0, 0, 200))
                selected_word_info['is_revealed'] = True
                selected_word_info['is_selected'] = True # Assuming you use this for something
                self.armed_word_idx = -1
                self.current_highlighted_for_dwell_idx = -1
                all_correct_found = all(w['is_revealed'] for w in self.word_grid if w['is_correct'])
                if all_correct_found:
                    self.display_feedback(f"All {self.current_mode} for '{self.current_target_word}'!", (0,200,200), duration=3)
                    if self.current_target_word in self.WORD_DATA: del self.WORD_DATA[self.current_target_word]
                    if not self.WORD_DATA: self.end_game("Congrats! All words done!")
                    else: self.setup_new_round()
        # --- End of Game Logic ---

        # --- Drawing (On the OpenCV frame, using self. for variables) ---
        processed_frame = frame # The frame MediaPipe worked on (already BGR)
        processed_frame[:] = (80, 50, 20) # Background

        # Header
        cv2.rectangle(processed_frame, (0,0), (self.FRAME_WIDTH_CAM, self.HEADER_HEIGHT), (50,30,10), -1)
        header_text = f"Find {self.current_mode.upper()} for: {self.current_target_word.upper()}"
        text_size_header, _ = cv2.getTextSize(header_text, self.FONT, 0.9, 2)
        cv2.putText(processed_frame, header_text, ((self.FRAME_WIDTH_CAM - text_size_header[0])//2, 40), self.FONT, 0.9, (200,200,255), 2)
        cv2.putText(processed_frame, f"Score: {self.score}", (10, 75), self.FONT, 0.7, (200,200,255), 2)
        text_size_timer, _ = cv2.getTextSize(f"Time: {time_remaining}", self.FONT, 0.7, 2)
        cv2.putText(processed_frame, f"Time: {time_remaining}", (self.FRAME_WIDTH_CAM - text_size_timer[0] - 10, 75), self.FONT, 0.7, (200,200,255), 2)
        cv2.putText(processed_frame, f"Mode: {self.current_mode.upper()} ('m' to switch)", (10, self.HEADER_HEIGHT - 10), self.FONT, 0.5, (150,150,200),1)

        # Word Grid
        for i, word_info in enumerate(self.word_grid):
            bg_color = (100, 70, 40); text_color = (220, 220, 220)
            if i == self.current_highlighted_for_dwell_idx and not word_info['is_revealed']:
                if i == self.armed_word_idx: bg_color = (180, 160, 80) # Armed
                else: # Dwelling
                    dwell_progress = min(1.0, (time.time() - self.dwell_start_time) / self.DWELL_TO_ARM_TIME)
                    base_col = np.array([100,70,40]); highlight_col = np.array([150,120,90])
                    bg_color = tuple(int(c) for c in (base_col + (highlight_col - base_col) * dwell_progress))
            if word_info['is_revealed']:
                if word_info['is_correct']: bg_color = (30, 150, 30) # Correct
                else: bg_color = (30, 30, 150) # Incorrect
            cv2.rectangle(processed_frame, (word_info['cell_x'], word_info['cell_y']), (word_info['cell_x'] + word_info['cell_w'], word_info['cell_y'] + word_info['cell_h']), bg_color, -1)
            cv2.rectangle(processed_frame, (word_info['cell_x'], word_info['cell_y']), (word_info['cell_x'] + word_info['cell_w'], word_info['cell_y'] + word_info['cell_h']), (180,150,120), 1)
            text_to_draw = word_info['display_text']
            text_size, _ = cv2.getTextSize(text_to_draw, self.FONT, self.FONT_SCALE_WORD, self.FONT_THICKNESS)
            text_x = word_info['x'] + (word_info['w'] - text_size[0]) // 2
            text_y = word_info['y'] + (word_info['h'] + text_size[1]) // 2
            cv2.putText(processed_frame, text_to_draw, (text_x, text_y), self.FONT, self.FONT_SCALE_WORD, text_color, self.FONT_THICKNESS)

        # Feedback Message
        if self.feedback_message and time.time() < self.feedback_end_time:
            text_size_feedback, _ = cv2.getTextSize(self.feedback_message, self.FONT, 0.8, 2)
            msg_x = (self.FRAME_WIDTH_CAM - text_size_feedback[0]) // 2
            msg_y = self.FRAME_HEIGHT_CAM - 30
            cv2.putText(processed_frame, self.feedback_message, (msg_x, msg_y), self.FONT, 0.8, self.feedback_color, 2)
        elif self.game_over: # Keep game over message persistent until reset
             text_size_feedback, _ = cv2.getTextSize(self.feedback_message, self.FONT, 1.0, 2)
             msg_x = (self.FRAME_WIDTH_CAM - text_size_feedback[0]) // 2
             msg_y = self.FRAME_HEIGHT_CAM // 2 # Center it
             cv2.putText(processed_frame, self.feedback_message, (msg_x, msg_y), self.FONT, 1.0, self.feedback_color, 2)


        # Pointer
        if not self.game_over:
            pointer_outline_color = (50, 255, 255)
            if self.armed_word_idx != -1: pointer_outline_color = (50, 180, 255) # Orange when armed
            cv2.circle(processed_frame, (self.pointer_x, self.pointer_y), self.pointer_radius, pointer_outline_color, 2)
        # --- End of Drawing ---

        self.display_cv_frame(processed_frame)

        # Frame timing analysis
        end_time = time.time()
        frame_time = (end_time - start_time) * 1000
        self.frame_times.append(frame_time)
        if len(self.frame_times) % 100 == 0:  # Log every 100 frames
            print(f"[Perf] Avg frame time: {np.mean(self.frame_times):.2f} ")

    def display_cv_frame(self, cv_frame):
        """Converts an OpenCV frame to QPixmap and displays it on the QLabel."""
        rgb_image = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        # Scale pixmap to fit label while maintaining aspect ratio, if label size is fixed
        # Or ensure label resizes with window and pixmap fills label
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))


    def start_game_or_reset(self):
        print("Brain Blinker: Starting game or resetting...")
        if self.cap is None:
            self.cap = cv2.VideoCapture(0) # Use your CAM_INDEX if different
            if not self.cap.isOpened():
                print("Error: Cannot open camera.")
                self.cap = None
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH_CAM)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT_CAM)

        if self.face_mesh is None:
             self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

        # Reset game state variables
        self.initial_nose_x = None
        self.initial_nose_y = None
        self.score = 0
        self.game_over = False
        self.feedback_message = ""
        self.current_mode = "synonyms" # Default mode
        # Potentially reload WORD_DATA if it was being depleted
        # For now, we assume it's static or managed by setup_new_round

        self.setup_new_round() # Start the first round

        self.game_running = True
        self.next_button.setEnabled(True)
        self.timer.start(30) # Update frame roughly 30 FPS (1000ms / 30 = ~33ms, use 30 for simplicity)
        self.setFocus() # Ensure widget can receive key events


    def stop_game_or_pause(self):
        print("Brain Blinker: Stopping game or pausing...")
        self.game_running = False
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.face_mesh is not None: # Close face_mesh if you re-initialize it often
            self.face_mesh.close() # This is important for MediaPipe
            self.face_mesh = None
        # Optionally, display a "Paused" or "Game Stopped" message on self.video_label


    def keyPressEvent(self, event):
        """Handle game-specific key presses."""
        if not self.game_running or self.game_over:
            # If game over and player presses 'n' or 'm', maybe restart?
            if self.game_over and (event.key() == Qt.Key_N or event.key() == Qt.Key_M):
                self.start_game_or_reset()
            return # Don't process game keys if not running or game over

        key = event.key()
        # Qt.Key_Q, Qt.Key_R, etc.
        if key == Qt.Key_R:
            self.initial_nose_x, self.initial_nose_y = None, None
            self.display_feedback("Head position reset!", (255,255,0))
        elif key == Qt.Key_M:
            self.current_mode = "antonyms" if self.current_mode == "synonyms" else "synonyms"
            self.display_feedback(f"Mode: {self.current_mode.upper()}", (255,255,0))
            if self.current_target_word in self.WORD_DATA: # Remove to ensure new word for new mode
                 # Check if WORD_DATA still has items before deleting
                if self.WORD_DATA and self.current_target_word in self.WORD_DATA:
                    del self.WORD_DATA[self.current_target_word] # This could empty WORD_DATA if not careful

            if not self.WORD_DATA : # If deleting emptied it, end game
                self.end_game("All words done!")
            else: # Otherwise setup new round for the new mode
                self.setup_new_round()

        elif key == Qt.Key_N:
            self.display_feedback(f"Skipping '{self.current_target_word}'", (255,165,0))
            # Similar logic for deleting from WORD_DATA
            if self.WORD_DATA and self.current_target_word in self.WORD_DATA:
                del self.WORD_DATA[self.current_target_word]
            if not self.WORD_DATA:
                self.end_game("All words done!")
            else:
                self.setup_new_round()
        # else: # If you want other keys to propagate to parent or default handling
        #     super().keyPressEvent(event)


    # Ensure to clean up when the widget is closed/hidden
    def closeEvent(self, event): # This is for QMainWindow, for QWidget use hideEvent
        self.stop_game_or_pause()
        super().closeEvent(event)

    def hideEvent(self, event): # Called when the widget is hidden (e.g. MainUI switches screen)
        self.stop_game_or_pause()
        super().hideEvent(event)

    def showEvent(self, event): # Called when the widget is shown
        # It's better to call start_game_or_reset from the MainUI's switch_screen method
        # to ensure it's called only when this screen specifically becomes active.
        # However, if you want it to auto-start when shown (e.g. if this is the initial widget)
        # self.start_game_or_reset()
        super().showEvent(event)

    def go_to_menu(self):
        self.back_to_menu.emit()

    def get_max_score(self):
        # Each correct selection is +10, so max score is 10 * total correct words shown
        total_correct = 0
        for word, data in self.WORD_DATA_ORIGINAL.items():
            mode = "synonyms" if self.current_mode == "synonyms" else "antonyms"
            total_correct += len(data[mode])
        return total_correct * 10

    

    def next_round(self):
        if self.WORD_DATA and self.current_target_word in self.WORD_DATA:
            del self.WORD_DATA[self.current_target_word]
        if not self.WORD_DATA:
            self.end_game("All words done!")
        else:
            self.setup_new_round()