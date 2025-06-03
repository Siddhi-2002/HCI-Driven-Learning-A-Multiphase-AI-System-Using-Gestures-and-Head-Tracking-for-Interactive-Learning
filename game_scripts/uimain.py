# ui_main.py
import sys, cv2, mediapipe as mp, numpy as np, time, math, random, locale
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QStackedWidget, QMainWindow,
                             QSizePolicy, QFrame, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPoint, QRect, QSize
from PyQt5.QtGui import QPainter, QColor, QFont, QPixmap, QImage, QBrush, QPen

# --- Import your logic classes ---
from aicrossword import CrosswordGame
from hand_gesture_game import HandGestureGame
from brainblinker import BrainBlinkerGame # Import your head tracking game

# Force Numpad decimal point to be '.'
locale.setlocale(locale.LC_NUMERIC, 'C')

class ConfettiParticle(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.color = QColor(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 200) # Bright, slightly transparent
        
        # Start from top, spread across width
        self.x_pos = random.randint(0, parent.width())
        self.y_pos = random.randint(-60, -10) # Start above the widget
        
        self.size = random.randint(8, 15)
        
        # Movement properties
        self.x_velocity = random.uniform(-1.5, 1.5) # Sideways drift
        self.y_velocity = random.uniform(2.0, 5.0)  # Downward speed
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-5, 5)

        self.setFixedSize(self.size * 2, self.size * 2) # QLabel size to encompass rotation
        self.move(int(self.x_pos), int(self.y_pos))
        self.show()

    def update_position(self, parent_height, parent_width):
        self.x_pos += self.x_velocity
        self.y_pos += self.y_velocity
        self.y_velocity += 0.15 # Gravity effect
        self.rotation += self.rotation_speed

        self.move(int(self.x_pos), int(self.y_pos))

        if self.y_pos > parent_height + self.size or \
           self.x_pos < -self.size*2 or self.x_pos > parent_width + self.size*2:
            return False
        return True


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.color)
        painter.setPen(Qt.NoPen)
        
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.rotation)
        painter.drawRect(-self.size // 2, -self.size // 4, self.size, self.size // 2)
        
# --- End of ConfettiParticle Class ---


# --- UI Configuration (mostly colors and fixed dimensions) ---
WINDOW_TITLE_HUB = "AI Game Hub"
WINDOW_TITLE_CROSSWORD = "AI Crossword Adventure!"
WINDOW_TITLE_GESTURE = "Hand Gesture Challenge!"
WINDOW_TITLE_BRAIN_BLINKER = "Brain Blinker: Tilt & Find!"
UI_WIDTH, UI_HEIGHT = 1280, 720

KEYBOARD_LAYOUT = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]
KEY_SIZE, KEY_SPACING = 55, 8
SPECIAL_BUTTONS_LAYOUT = [("NEW GAME", "NEW"), ("HINT", "HINT"), ("DEL", "DEL"), ("ENTER", "ENTER"), ("NEXT CLUE", "NEXT")]
SPECIAL_BTN_WIDTH, SPECIAL_BTN_HEIGHT = KEY_SIZE * 1.9, KEY_SIZE
DWELL_TIME_THRESHOLD = 1.0
FEEDBACK_MESSAGE_DURATION = 3.0
PREVIEW_WIDTH, PREVIEW_HEIGHT = 280, 157

# --- UI Colors (QColor) ---
PREVIEW_BORDER_COLOR_QT = QColor(100, 100, 100)
COLOR_BACKGROUND_QT = QColor(230, 240, 255)
COLOR_TITLE_TEXT_QT = QColor(255, 105, 180)
COLOR_SCORE_TEXT_QT = QColor(0, 100, 0)
COLOR_TEXT_LIGHT_QT = QColor(240, 240, 240)
COLOR_TEXT_DARK_QT = QColor(50, 50, 50)
COLOR_GRID_CELL_ACTIVE_QT = QColor(255, 255, 255)
COLOR_GRID_CELL_INACTIVE_QT = QColor(180, 200, 220)
COLOR_GRID_BORDER_QT = QColor(100, 130, 160)
COLOR_GRID_TEXT_NORMAL_QT = COLOR_TEXT_DARK_QT
COLOR_GRID_TEXT_CORRECT_QT = QColor(0, 180, 0)
COLOR_GRID_TEXT_HINTED_QT = QColor(0, 0, 200)
COLOR_GRID_TEXT_REVEALED_QT = QColor(255, 100, 0)
COLOR_GRID_HIGHLIGHT_CURRENT_CELL_QT = QColor(255, 165, 0)
COLOR_GRID_HIGHLIGHT_CURRENT_WORD_QT = QColor(173, 216, 230, 180)
COLOR_GRID_CLUE_NUM_QT = QColor(120, 120, 120)
COLOR_KEY_ROW_BGS_QT = [QColor(150, 70, 150), QColor(120, 50, 120), QColor(100, 30, 100)]
COLOR_KEY_HOVER_BG_FACTOR = 1.25
COLOR_KEY_SPECIAL_BG_QT = QColor(180, 100, 180)
COLOR_KEY_SPECIAL_HOVER_BG_QT = QColor(200, 130, 200)
COLOR_KEY_TEXT_QT = COLOR_TEXT_LIGHT_QT
COLOR_KEY_SELECTED_BG_QT = QColor(255, 255, 0)
COLOR_CURSOR_QT = QColor(255, 0, 255)
COLOR_CURSOR_BORDER_QT = QColor(255, 255, 255)
COLOR_INSTRUCTION_TEXT_QT = QColor(80, 80, 150)
COLOR_FEEDBACK_GOOD_QT = QColor(0, 150, 0)
COLOR_FEEDBACK_BAD_QT = QColor(200, 0, 0)


# ============================================
# Camera Thread (For Hand Gesture & Crossword Cursor)
# ============================================
class CameraThread(QThread):
    frameReady = pyqtSignal(np.ndarray)
    handDataReady = pyqtSignal(object, QPoint)
    def __init__(self, parent=None, webcam_id=0):
        super().__init__(parent)
        self.running = False
        self.hands = None
        self.cap = None
        self.cursor_pos = QPoint(0,0)
        self.WEBCAM_ID = webcam_id
        self.last_smoothed_tip_x, self.last_smoothed_tip_y = None, None
        self.CURSOR_SMOOTHING_ALPHA, self.MOVEMENT_THRESHOLD = 0.50, 2

    def run(self):
        self.running = True
        print("CameraThread: starting run()...")
        self.cap = None
        self.hands = None
        try:
            print(f"CameraThread: Attempting to open webcam {self.WEBCAM_ID}...")
            self.cap = cv2.VideoCapture(self.WEBCAM_ID)
            if not self.cap.isOpened():
                print(f"CameraThread: Error - Webcam {self.WEBCAM_ID} not found or already in use.")
                return 

            print(f"CameraThread: Webcam {self.WEBCAM_ID} opened successfully.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            
            mp_hands_sol = mp.solutions.hands 
            self.hands = mp_hands_sol.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
            print("CameraThread: MediaPipe Hands initialized.")

            while self.running: 
                if not self.cap or not self.cap.isOpened():
                    print("CameraThread: Camera capture became invalid or closed unexpectedly.")
                    break 

                success, raw_cam_frame = self.cap.read()
                if not success:
                    time.sleep(0.01) 
                    continue

                raw_cam_frame = cv2.flip(raw_cam_frame, 1)
                self.frameReady.emit(raw_cam_frame) 

                results = None
                hand_landmarks_for_active_hand = None
                try:
                    image_for_mp = cv2.cvtColor(raw_cam_frame, cv2.COLOR_BGR2RGB)
                    image_for_mp.flags.writeable = False 
                    results = self.hands.process(image_for_mp)
                except Exception as hand_e:
                    print(f"CameraThread: Hand processing error: {hand_e}")

                current_cursor_pos = QPoint(-1, -1) 
                if results and results.multi_hand_landmarks:
                    hand_landmarks_for_active_hand = results.multi_hand_landmarks[0]
                    tip = hand_landmarks_for_active_hand.landmark[mp_hands_sol.HandLandmark.INDEX_FINGER_TIP]
                    tip_x_raw, tip_y_raw = int(tip.x * UI_WIDTH), int(tip.y * UI_HEIGHT)
                    
                    if self.last_smoothed_tip_x is None or self.last_smoothed_tip_y is None:
                        sx, sy = tip_x_raw, tip_y_raw
                    else:
                        sx = int(self.CURSOR_SMOOTHING_ALPHA * tip_x_raw + (1 - self.CURSOR_SMOOTHING_ALPHA) * self.last_smoothed_tip_x)
                        sy = int(self.CURSOR_SMOOTHING_ALPHA * tip_y_raw + (1 - self.CURSOR_SMOOTHING_ALPHA) * self.last_smoothed_tip_y)
                        if abs(sx - self.last_smoothed_tip_x) < self.MOVEMENT_THRESHOLD: sx = self.last_smoothed_tip_x
                        if abs(sy - self.last_smoothed_tip_y) < self.MOVEMENT_THRESHOLD: sy = self.last_smoothed_tip_y
                    
                    self.last_smoothed_tip_x, self.last_smoothed_tip_y = sx, sy
                    current_cursor_pos = QPoint(sx, sy)
                
                self.handDataReady.emit(hand_landmarks_for_active_hand, current_cursor_pos)
                QThread.msleep(10) 

        except Exception as e:
            print(f"CameraThread: Unexpected error in run loop: {e}")
        finally:
            print("CameraThread: run() method finishing. Cleaning up...")
            if self.cap:
                self.cap.release()
                self.cap = None
                print("CameraThread: Webcam released.")
            if self.hands:
                self.hands.close()
                self.hands = None
                print("CameraThread: MediaPipe Hands closed.")
            self.running = False 
            print("CameraThread: run() finished and cleaned up.")

    def stop(self):
        print("CameraThread: stop() called, requesting thread to terminate.")
        self.running = False

# ============================================
# Crossword Game Widget
# ============================================
class CrosswordGameWidget(QWidget):
    keyPressedSignal = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); self.setFocusPolicy(Qt.StrongFocus)
        self.PUZZLE, self.solution_grid, self.player_grid, self.player_grid_cell_status = [], [], [], []
        self.active_cells, self.clue_numbers_map, self.current_entry_cells = [], {}, []
        self.GRID_ROWS, self.GRID_COLS, self.CELL_SIZE = 10, 10, 38
        self.GRID_START_X, self.GRID_START_Y, self.KEYBOARD_START_Y = 50, 60, 0
        self.INSTRUCTION_TEXT_Y_POS, self.CLUE_TEXT_START_Y = 0, 0
        self.SCORE, self.current_clue_idx, self.current_selected_cell_idx_in_entry = 0, 0, 0
        self.game_over_message, self.feedback_message = "", ""
        self.solved_word_indices = set()
        self.cursor_pos = QPoint(-1, -1); self.key_rects = {}
        self.hovered_key_char, self.selected_key_char_display = None, None
        self.last_hovered_key_char, self.hover_start_time = None, None
        self.feedback_message_color = COLOR_FEEDBACK_GOOD_QT
        self.feedback_timer = QTimer(self); self.feedback_timer.setSingleShot(True); self.feedback_timer.timeout.connect(self.clear_feedback)
        self.selection_flash_timer = QTimer(self); self.selection_flash_timer.setSingleShot(True); self.selection_flash_timer.timeout.connect(self.clear_selection_flash); self.selection_flash_timer.setInterval(300)
        self.fonts = {'title':QFont("Arial",26,QFont.Bold),'score':QFont("Arial",12,QFont.Bold),'instr':QFont("Arial",9),'feedback':QFont("Arial",12,QFont.Bold),'clue':QFont("Arial",10),'grid_char':QFont("Arial",14,QFont.Bold),'grid_num':QFont("Arial",7),'key_char':QFont("Arial",16,QFont.Bold),'key_special':QFont("Arial",9)}
        
        self.confetti_particles = []
        self.confetti_animation_timer = QTimer(self) 
        self.confetti_animation_timer.timeout.connect(self.update_confetti_animation)
        self.confetti_spawn_timer = QTimer(self) 
        self.confetti_spawn_timer.timeout.connect(self.spawn_confetti_wave)
        self.confetti_duration_timer = QTimer(self) 
        self.confetti_duration_timer.setSingleShot(True)
        self.confetti_duration_timer.timeout.connect(self.stop_confetti_animation_after_duration) 
        self.is_reward_active = False


    def handle_game_completion(self, score): 
          if not self.is_reward_active:
              print("CrosswordWidget: Game complete! Starting sustained confetti.")
              self.is_reward_active = True
              self.show_feedback(f"CONGRATULATIONS! Score: {score}", QColor("gold"))
              self.start_sustained_confetti()

    def start_sustained_confetti(self):
          if self.confetti_animation_timer.isActive(): 
              return
          self.clear_confetti() 
          self.confetti_animation_timer.start(30) 
          self.spawn_confetti_wave() 
          self.confetti_spawn_timer.start(250) 
          self.confetti_duration_timer.start(5000) 

    def spawn_confetti_wave(self):
          if not self.isVisible():
              self.stop_all_confetti_activity()
              return
          num_to_spawn_this_wave = random.randint(20, 35)
          for _ in range(num_to_spawn_this_wave):
              if len(self.confetti_particles) < 250: 
                  particle = ConfettiParticle(self) 
                  self.confetti_particles.append(particle)
              else:
                  break 

    def update_confetti_animation(self):
          active_particles = []
          parent_h = self.height()
          parent_w = self.width()
          for particle in self.confetti_particles:
              if hasattr(particle, 'update_position') and callable(getattr(particle, 'update_position')):
                  if particle.update_position(parent_h, parent_w): 
                      active_particles.append(particle)
                  else:
                      particle.deleteLater()
              else: 
                  print("Error: particle does not have update_position method") 
                  particle.deleteLater() 
          self.confetti_particles = active_particles
          if not self.confetti_spawn_timer.isActive() and not self.confetti_particles:
              self.confetti_animation_timer.stop()

    def stop_confetti_animation_after_duration(self):
          print("CrosswordWidget: Confetti duration ended, stopping spawn.")
          self.confetti_spawn_timer.stop()

    def stop_all_confetti_activity(self):
          print("CrosswordWidget: Stopping ALL confetti activity.")
          self.confetti_animation_timer.stop()
          self.confetti_spawn_timer.stop()
          self.confetti_duration_timer.stop()
          self.clear_confetti()

    def clear_confetti(self):
        for particle in self.confetti_particles:
            particle.deleteLater()
        self.confetti_particles.clear()

    def reset_for_new_game(self): 
        print("CrosswordWidget: Resetting for new game, stopping rewards.")
        self.stop_all_confetti_activity() 
        self.is_reward_active = False 

    def update_game_state(self, state_dict):
        for key, value in state_dict.items():
            if hasattr(self, key): setattr(self, key, value)
        self.update()
    def set_cursor_pos(self, pos):
        if pos != self.cursor_pos: self.cursor_pos = pos; self.check_hover(); self.update()
    
    def show_feedback(self, message, is_good_color_or_qcolor): 
        if isinstance(is_good_color_or_qcolor, QColor):
            self.feedback_message_color = is_good_color_or_qcolor
        else: 
             self.feedback_message_color = COLOR_FEEDBACK_GOOD_QT if is_good_color_or_qcolor else COLOR_FEEDBACK_BAD_QT
        self.feedback_message = message
        self.feedback_timer.stop();
        if message: self.feedback_timer.start(int(FEEDBACK_MESSAGE_DURATION * 1000))
        self.update()

    def clear_feedback(self): self.feedback_message = ""; self.update()
    def show_selection_flash(self, key_char):
        if key_char!="NEW": self.selected_key_char_display=key_char; self.selection_flash_timer.start(); self.update()
    def clear_selection_flash(self): self.selected_key_char_display=None; self.update()
    def setup_layout_dimensions(self):
        w, h = self.width(), self.height()
        KEYBOARD_BLOCK_H = (3 * (KEY_SIZE + KEY_SPACING)) + (SPECIAL_BTN_HEIGHT + KEY_SPACING)
        INSTR_AREA_H, BOTTOM_MARGIN, TITLE_AREA_TOP_MARGIN = 35, 20, 60
        self.KEYBOARD_START_Y = h - (KEYBOARD_BLOCK_H + INSTR_AREA_H + BOTTOM_MARGIN) + INSTR_AREA_H
        self.INSTRUCTION_TEXT_Y_POS = self.KEYBOARD_START_Y - 15
        self.GRID_START_Y = TITLE_AREA_TOP_MARGIN + 10
        available_grid_h = self.KEYBOARD_START_Y - self.GRID_START_Y - 80
        available_grid_w = w - 40
        if self.GRID_ROWS > 0 and self.GRID_COLS > 0:
            cell_h = available_grid_h // self.GRID_ROWS if available_grid_h > 0 else 30
            cell_w = available_grid_w // self.GRID_COLS if available_grid_w > 0 else 30
            self.CELL_SIZE = max(18, min(cell_h, cell_w, 45))
        else: self.CELL_SIZE = 30
        self.GRID_TOTAL_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_TOTAL_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_START_X = max(20, (w - self.GRID_TOTAL_WIDTH) // 2)
        self.CLUE_TEXT_START_Y = self.GRID_START_Y + self.GRID_TOTAL_HEIGHT + 15

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        w = self.width(); painter.fillRect(self.rect(), COLOR_BACKGROUND_QT)
        self.setup_layout_dimensions()
        painter.setFont(self.fonts['title']); painter.setPen(COLOR_TITLE_TEXT_QT)
        painter.drawText(QRect(0, 5, w, 50), Qt.AlignCenter | Qt.AlignTop, WINDOW_TITLE_CROSSWORD)
        painter.setFont(self.fonts['score']); painter.setPen(COLOR_SCORE_TEXT_QT)
        painter.drawText(QRect(w - 160, 25, 150, 30), Qt.AlignRight | Qt.AlignVCenter, f"Score: {self.SCORE}")
        self._draw_crossword_grid(painter); self._draw_clue(painter, w)
        self.key_rects = self._draw_keyboard(painter, w)
        painter.setFont(self.fonts['instr']); painter.setPen(COLOR_INSTRUCTION_TEXT_QT)
        painter.drawText(QRect(0,self.INSTRUCTION_TEXT_Y_POS-10,w,20), Qt.AlignCenter, "Point & Dwell. ESC=Menu. R=New.")
        if self.feedback_message:
            painter.setFont(self.fonts['feedback']);painter.setPen(self.feedback_message_color)
            painter.drawText(QRect(0,self.INSTRUCTION_TEXT_Y_POS-35,w,30), Qt.AlignCenter, self.feedback_message)
        if self.cursor_pos.x() >= 0:
             painter.setPen(Qt.NoPen); painter.setBrush(QBrush(COLOR_CURSOR_QT))
             painter.drawEllipse(self.cursor_pos, 8, 8)
             painter.setPen(QPen(COLOR_CURSOR_BORDER_QT, 1)); painter.setBrush(Qt.NoBrush)
             painter.drawEllipse(self.cursor_pos, 9, 9)
        painter.end()

    def _draw_crossword_grid(self, painter):
        if not self.active_cells or not self.player_grid or not self.GRID_ROWS or not self.GRID_COLS : return
        grid_pen=QPen(COLOR_GRID_BORDER_QT,1); hl_w_pen=QPen(COLOR_GRID_HIGHLIGHT_CURRENT_WORD_QT,2); hl_c_pen=QPen(COLOR_GRID_HIGHLIGHT_CURRENT_CELL_QT,3)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if not(0<=r<len(self.active_cells)and 0<=c<len(self.active_cells[0])and 0<=r<len(self.player_grid)and 0<=c<len(self.player_grid[0])and 0<=r<len(self.player_grid_cell_status)and 0<=c<len(self.player_grid_cell_status[0])):continue
                x1,y1=self.GRID_START_X+c*self.CELL_SIZE,self.GRID_START_Y+r*self.CELL_SIZE;cell_rect=QRect(x1,y1,self.CELL_SIZE,self.CELL_SIZE)
                if self.active_cells[r][c]:
                    painter.setBrush(QBrush(COLOR_GRID_CELL_ACTIVE_QT));painter.setPen(grid_pen);painter.drawRect(cell_rect)
                    if(r,c)in self.clue_numbers_map:painter.setFont(self.fonts['grid_num']);painter.setPen(COLOR_GRID_CLUE_NUM_QT);painter.drawText(QRect(x1+2,y1+1,self.CELL_SIZE-4,10),Qt.AlignLeft|Qt.AlignTop,self.clue_numbers_map[(r,c)])
                    l=self.player_grid[r][c];s=self.player_grid_cell_status[r][c];tc=COLOR_GRID_TEXT_NORMAL_QT
                    if s=='correct':tc=COLOR_GRID_TEXT_CORRECT_QT
                    elif s=='hinted':tc=COLOR_GRID_TEXT_HINTED_QT
                    elif s=='revealed':tc=COLOR_GRID_TEXT_REVEALED_QT
                    if l and l!=' ':painter.setFont(self.fonts['grid_char']);painter.setPen(tc);painter.drawText(cell_rect,Qt.AlignCenter,l)
                else:painter.setBrush(QBrush(COLOR_GRID_CELL_INACTIVE_QT));painter.setPen(Qt.NoPen);painter.drawRect(cell_rect)
        painter.setBrush(Qt.NoBrush);painter.setPen(hl_w_pen)
        for re,ce in self.current_entry_cells:
            if 0<=re<self.GRID_ROWS and 0<=ce<self.GRID_COLS:painter.drawRect(self.GRID_START_X+ce*self.CELL_SIZE,self.GRID_START_Y+re*self.CELL_SIZE,self.CELL_SIZE,self.CELL_SIZE)
        sc=self.get_current_selected_cell()
        if sc:
            rs,cs=sc
            if 0<=rs<self.GRID_ROWS and 0<=cs<self.GRID_COLS:painter.setPen(hl_c_pen);painter.drawRect(self.GRID_START_X+cs*self.CELL_SIZE,self.GRID_START_Y+rs*self.CELL_SIZE,self.CELL_SIZE,self.CELL_SIZE)
    
    def _draw_clue(self, painter, effective_width):
        if not self.PUZZLE or not(0<=self.current_clue_idx<len(self.PUZZLE)):return
        p=self.PUZZLE[self.current_clue_idx];di="Across"if p["direction"]==0 else"Down"
        ct=f"{p.get('clue_num',self.current_clue_idx+1)}. {di}: {p['clue']}"
        painter.setFont(self.fonts['clue']);painter.setPen(COLOR_TEXT_DARK_QT)
        mcw=effective_width-self.GRID_START_X-20;cr=QRect(self.GRID_START_X,self.CLUE_TEXT_START_Y,mcw,60)
        painter.drawText(cr,Qt.AlignLeft|Qt.TextWordWrap,ct)
    
    def _draw_keyboard(self, painter, effective_width):
        lkr={};y=self.KEYBOARD_START_Y;bp=QPen(COLOR_TEXT_DARK_QT,1)
        for ri,rk in enumerate(KEYBOARD_LAYOUT):
            rw=len(rk)*KEY_SIZE+(len(rk)-1)*KEY_SPACING;xs=(effective_width-rw)//2
            bbg=COLOR_KEY_ROW_BGS_QT[ri%len(COLOR_KEY_ROW_BGS_QT)]
            for ci,kc in enumerate(rk):
                x=xs+ci*(KEY_SIZE+KEY_SPACING);r=QRect(x,y,KEY_SIZE,KEY_SIZE);lkr[kc]=r;bg=QColor(bbg)
                if self.hovered_key_char==kc:h,s,v,a=bg.getHsv();v=min(255,int(v*COLOR_KEY_HOVER_BG_FACTOR));bg.setHsv(h,s,v,a)
                if self.selected_key_char_display==kc:bg=COLOR_KEY_SELECTED_BG_QT
                painter.setBrush(QBrush(bg));painter.setPen(bp);painter.drawRect(r)
                painter.setFont(self.fonts['key_char']);painter.setPen(COLOR_KEY_TEXT_QT);painter.drawText(r,Qt.AlignCenter,kc)
            y+=KEY_SIZE+KEY_SPACING
        ns=len(SPECIAL_BUTTONS_LAYOUT);tsw=ns*SPECIAL_BTN_WIDTH+(ns-1)*KEY_SPACING;xss=(effective_width-tsw)//2;cx=xss
        for bt,bk in SPECIAL_BUTTONS_LAYOUT:
            r=QRect(int(cx),int(y),int(SPECIAL_BTN_WIDTH),int(SPECIAL_BTN_HEIGHT));lkr[bk]=r;bg=QColor(COLOR_KEY_SPECIAL_BG_QT)
            if self.hovered_key_char==bk:bg=COLOR_KEY_SPECIAL_HOVER_BG_QT
            if self.selected_key_char_display==bk:bg=COLOR_KEY_SELECTED_BG_QT
            painter.setBrush(QBrush(bg));painter.setPen(bp);painter.drawRect(r)
            painter.setFont(self.fonts['key_special']);painter.setPen(COLOR_KEY_TEXT_QT);painter.drawText(r,Qt.AlignCenter,bt)
            cx+=SPECIAL_BTN_WIDTH+KEY_SPACING
        return lkr
    
    def check_hover(self):
        ch=None;lcp=self.cursor_pos
        if lcp.x()>=0:
            for k,r in self.key_rects.items():
                if r.contains(lcp):ch=k;break
        ai=(self.game_over_message==""or ch=="NEW")
        if ch and ai:
            if ch==self.last_hovered_key_char:
                if self.hover_start_time is None:self.hover_start_time=time.time()
                elif time.time()-self.hover_start_time>=DWELL_TIME_THRESHOLD:
                    self.keyPressedSignal.emit(ch);self.show_selection_flash(ch)
                    self.hover_start_time,self.last_hovered_key_char,ch=None,None,None
            else:self.last_hovered_key_char,self.hover_start_time=ch,time.time()
        else:self.last_hovered_key_char,self.hover_start_time=None,None
        nhc=ch if ai else None
        if nhc!=self.hovered_key_char:self.hovered_key_char=nhc;self.update()
    
    def get_current_selected_cell(self):
        if self.current_entry_cells and 0 <= self.current_selected_cell_idx_in_entry < len(self.current_entry_cells):
            return self.current_entry_cells[self.current_selected_cell_idx_in_entry]
        return None

# ============================================
# Main Application Window
# ============================================
class MainApp(QMainWindow):
    MENU_BUTTON_STYLE = """QPushButton{{font-size:24px;font-weight:bold;color:white;background-color:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {color1},stop:1 {color2});border:2px solid {color2};border-radius:15px;padding:25px;min-height:100px}}QPushButton:hover{{background-color:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {hover1},stop:1 {hover2})}}QPushButton:pressed{{background-color:{color2}}}"""
    BACK_BTN_STYLE = """
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
"""
    def __init__(self):
        super().__init__(); self.setWindowTitle(WINDOW_TITLE_HUB); self.setFixedSize(UI_WIDTH, UI_HEIGHT)
        self.crossword_engine = CrosswordGame()
        self.gesture_engine = HandGestureGame() # Retain for potential future direct use, but HGG widget handles itself
        self.brain_blinker_engine = None # BrainBlinkerGame is the widget

        self.is_game_active = False; self.active_game_type = None
        self._init_ui_components()
        self._init_camera_system() 

    def _init_ui_components(self):
        self.stacked_widget = QStackedWidget(self); self.setCentralWidget(self.stacked_widget)
        self.main_menu_screen = self._build_main_menu_screen()
        self.crossword_game_screen = self._build_crossword_game_screen()
        self.gesture_game_screen = self._build_gesture_game_screen()
        self.brain_blinker_game_screen = self._build_brain_blinker_game_screen()

        self.stacked_widget.addWidget(self.main_menu_screen)
        self.stacked_widget.addWidget(self.crossword_game_screen)
        self.stacked_widget.addWidget(self.gesture_game_screen)
        self.stacked_widget.addWidget(self.brain_blinker_game_screen)
        self.stacked_widget.setCurrentWidget(self.main_menu_screen)
        
        if hasattr(self.crossword_engine, 'game_completed_signal') and \
           hasattr(self.crossword_display_widget, 'handle_game_completion'):
            self.crossword_engine.game_completed_signal.connect(self.crossword_display_widget.handle_game_completion)

    def _build_main_menu_screen(self):
        screen = QWidget(); layout = QVBoxLayout(screen)
        layout.setContentsMargins(100,50,100,50); layout.setSpacing(30) 
        title = QLabel("AI Game Hub"); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel{font-size:52px;font-weight:bold;color:#2C3E50;padding-bottom:30px}")
        layout.addWidget(title)

        btn_c = QPushButton("AI Crossword Challenge"); btn_c.setStyleSheet(self.MENU_BUTTON_STYLE.format(color1="#8E44AD",color2="#5E3370",hover1="#A569BD",hover2="#7D3C98")); btn_c.clicked.connect(self.launch_crossword_game) 
        layout.addWidget(btn_c)

        btn_g = QPushButton("Gesture Quest"); btn_g.setStyleSheet(self.MENU_BUTTON_STYLE.format(color1="#27AE60",color2="#1E8449",hover1="#2ECC71",hover2="#239B56")); btn_g.clicked.connect(self.launch_gesture_game) 
        layout.addWidget(btn_g)

        btn_bb = QPushButton("Brain Blinker: Tilt & Find"); btn_bb.setStyleSheet(self.MENU_BUTTON_STYLE.format(color1="#E67E22",color2="#D35400",hover1="#F39C12",hover2="#BA4A00")); btn_bb.clicked.connect(self.launch_brain_blinker_game) 
        layout.addWidget(btn_bb)

        layout.addStretch(1)
        return screen

    def _build_crossword_game_screen(self):
        screen = QWidget(); main_layout = QHBoxLayout(screen)
        main_layout.setContentsMargins(10,10,10,10); main_layout.setSpacing(10)
        self.crossword_display_widget = CrosswordGameWidget(self)
        self.crossword_display_widget.keyPressedSignal.connect(self.process_crossword_input)
        main_layout.addWidget(self.crossword_display_widget, stretch=4)
        sidebar = QWidget(); sidebar_layout = QVBoxLayout(sidebar); sidebar.setFixedWidth(PREVIEW_WIDTH+20)
        sidebar_layout.setContentsMargins(0,0,0,0); sidebar_layout.setSpacing(15)
        self.camera_preview_label = QLabel("Camera Off"); self.camera_preview_label.setFixedSize(PREVIEW_WIDTH,PREVIEW_HEIGHT); self.camera_preview_label.setAlignment(Qt.AlignCenter); self.camera_preview_label.setStyleSheet(f"QLabel{{background-color:black;color:white;border:2px solid {PREVIEW_BORDER_COLOR_QT.name()}}}")
        sidebar_layout.addWidget(self.camera_preview_label, alignment=Qt.AlignTop|Qt.AlignHCenter)
        btn_back = QPushButton("<< Back to Menu")
        btn_back.setStyleSheet(self.BACK_BTN_STYLE)  
        btn_back.clicked.connect(self.return_to_main_menu)
        sidebar_layout.addWidget(btn_back, alignment=Qt.AlignTop|Qt.AlignHCenter)
        sidebar_layout.addStretch(1)
        main_layout.addWidget(sidebar, stretch=1)
        return screen

    def _build_gesture_game_screen(self):
        self.gesture_game_widget_instance = HandGestureGame(self) 
        self.gesture_game_widget_instance.back_to_main_menu_signal.connect(self.return_to_main_menu)
        return self.gesture_game_widget_instance 

    def _build_brain_blinker_game_screen(self):
        screen = BrainBlinkerGame(self) # This is the QWidget
        screen.back_to_menu.connect(self.return_to_main_menu)
        return screen

    def _init_camera_system(self):
        self.camera_thread = CameraThread(self, webcam_id=0)
        self.camera_thread.frameReady.connect(self.update_camera_previews)
        self.camera_thread.handDataReady.connect(self.handle_hand_data_from_thread)
        # DO NOT START THE THREAD HERE. It will be started by games that need it.
        print("MainApp: Shared CameraThread initialized (but not started).")

    # --- Game Launchers & Menu Navigation ---
    def return_to_main_menu(self):
        current_active_widget = self.stacked_widget.currentWidget()
        current_game_type = self.active_game_type 
        self.setWindowTitle(WINDOW_TITLE_HUB)

        if self.is_game_active:
            if current_game_type == "crossword":
                if self.camera_thread.isRunning():
                    print(f"MainApp: Stopping shared CameraThread as {current_game_type} is ending.")
                    self.camera_thread.stop()
                    if not self.camera_thread.wait(1500): 
                        print("MainApp: Shared CameraThread did not stop gracefully, attempting to terminate.")
                        self.camera_thread.terminate() 
                        self.camera_thread.wait() 
                    else:
                        print("MainApp: Shared CameraThread stopped successfully.")
                self.camera_preview_label.setText("Camera Off")
                self.camera_thread.last_smoothed_tip_x = None
                self.camera_thread.last_smoothed_tip_y = None

            if hasattr(current_active_widget, 'stop_game_or_pause'):
                print(f"MainApp: Calling stop_game_or_pause for {current_game_type} widget.")
                current_active_widget.stop_game_or_pause()

            self.is_game_active = False
            self.active_game_type = None

        self.stacked_widget.setCurrentWidget(self.main_menu_screen)
        print("MainApp: Returned to main menu.")

    def launch_crossword_game(self):
        self.setWindowTitle(WINDOW_TITLE_CROSSWORD)
        self.active_game_type = "crossword"
        self.is_game_active = True
        self.stacked_widget.setCurrentWidget(self.crossword_game_screen)
        self.crossword_engine.start_new_game() # Logic engine start
        if hasattr(self.crossword_display_widget, 'reset_for_new_game'):
            self.crossword_display_widget.reset_for_new_game()

        if not self.camera_thread.isRunning():
            print("MainApp: Starting shared CameraThread for Crossword game.")
            self.camera_thread.start() 
        else:
            # This case implies a logic error in camera management if it happens.
            print("MainApp: Shared CameraThread was already running when launching Crossword. This might indicate an issue.")
            # Attempt to stop and restart to ensure a clean state.
            self.camera_thread.stop()
            if self.camera_thread.wait(500):
                 print("MainApp: Restarting shared camera thread for Crossword.")
                 self.camera_thread.start()
            else:
                 print("MainApp: Could not stop existing camera thread to restart for Crossword. Camera might not work.")


        self._update_crossword_ui_from_engine()
        QTimer.singleShot(50, lambda: self.crossword_display_widget.setFocus())

    def launch_gesture_game(self):
        self.setWindowTitle(WINDOW_TITLE_GESTURE)
        self.active_game_type = "gesture"
        self.stacked_widget.setCurrentWidget(self.gesture_game_widget_instance) 

        if not self.is_game_active:
            self.is_game_active = True
        
        if hasattr(self.gesture_game_widget_instance, 'start_game_or_reset'):
            print("MainApp: Starting HandGestureGame.")
            self.gesture_game_widget_instance.start_game_or_reset() 
        
        self.gesture_game_widget_instance.setFocus() 

    def launch_brain_blinker_game(self):
        self.setWindowTitle(WINDOW_TITLE_BRAIN_BLINKER)
        self.active_game_type = "brain_blinker"
        self.stacked_widget.setCurrentWidget(self.brain_blinker_game_screen)
        if not self.is_game_active: 
            self.is_game_active = True
        
        if hasattr(self.brain_blinker_game_screen, 'start_game_or_reset'):
            print("MainApp: Starting BrainBlinkerGame.")
            self.brain_blinker_game_screen.start_game_or_reset()
        self.brain_blinker_game_screen.setFocus() 

    # --- Camera/Hand Data Handling ---
    def update_camera_previews(self, cv_img):
        if not self.is_game_active: return
        
        # Only update preview for Crossword, as other games manage their own video display
        if self.active_game_type != "crossword": 
            # If the camera thread is somehow running when it shouldn't be, stop it.
            # This is a safeguard, ideally launch/return logic handles this.
            # if self.camera_thread.isRunning():
            #     print("MainApp: Safeguard - Stopping shared camera thread as non-crossword game is active.")
            #     self.camera_thread.stop()
            #     self.camera_thread.wait(500)
            return
        
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB); h,w,ch = rgb_image.shape
            q_img = QImage(rgb_image.data, w, h, ch*w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            if self.active_game_type == "crossword" and self.camera_preview_label: # Check label existence
                scaled_pixmap = pixmap.scaled(PREVIEW_WIDTH, PREVIEW_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.camera_preview_label.setPixmap(scaled_pixmap)
        except Exception as e: print(f"Preview update error: {e}")


    def handle_hand_data_from_thread(self, hand_landmarks, smoothed_cursor_pos):
        if not self.is_game_active: return
        
        # Only process hand data if the active game is Crossword
        if self.active_game_type != "crossword": return
        
        if self.crossword_display_widget: # Check widget existence
            try:
                # mapFromGlobal might not be needed if smoothed_cursor_pos is already in UI_WIDTH/HEIGHT scale
                # For now, assuming it's screen-global and needs mapping to the specific widget.
                # Test this carefully. If cursor seems off, direct use of smoothed_cursor_pos might be better
                # if the CameraThread calculates it based on UI_WIDTH/UI_HEIGHT.
                widget_cursor_pos = self.crossword_display_widget.mapFromGlobal(self.mapToGlobal(smoothed_cursor_pos))
                self.crossword_display_widget.set_cursor_pos(widget_cursor_pos)
            except Exception as e:
                print(f"Error mapping/setting cursor pos for crossword: {e}")
                # Fallback or direct use if mapping fails (less accurate if widget is not full screen)
                # self.crossword_display_widget.set_cursor_pos(smoothed_cursor_pos)


    # --- UI Update Methods (from Logic Engines) ---
    def _update_crossword_ui_from_engine(self):
        if not (self.is_game_active and self.active_game_type == "crossword" and self.crossword_display_widget): return
        state = {attr: getattr(self.crossword_engine, attr) for attr in [
            "PUZZLE", "GRID_ROWS", "GRID_COLS", "solution_grid", "player_grid",
            "player_grid_cell_status", "active_cells", "clue_numbers_map", "current_clue_idx",
            "current_entry_cells", "current_selected_cell_idx_in_entry", "game_over_message",
            "solved_word_indices", "SCORE", "CELL_SIZE"
        ] if hasattr(self.crossword_engine, attr)}
        self.crossword_display_widget.update_game_state(state)
        fb_info = self.crossword_engine.feedback_message_info
        self.crossword_display_widget.show_feedback(fb_info["text"], QColor("green") if fb_info["is_good"] else QColor("red")) 
        if not self.crossword_engine.game_over_message:
            if self.crossword_engine.check_game_completion(): # This will emit the signal
                 # The signal handler in CrosswordGameWidget will show its own "Game Over" message
                 pass


    def _update_gesture_ui_from_engine(self):
        # This method is less relevant now as HandGestureGame is a self-contained QWidget
        # However, if you had a separate logic engine and UI elements in MainApp for gesture, it would be used.
        # For now, HandGestureGame updates its own UI.
        pass


    # --- Input Processing (Delegates to active game engine) ---
    def process_crossword_input(self, key_char):
        if self.active_game_type == "crossword":
            self.crossword_engine.process_key_press(key_char)
            self._update_crossword_ui_from_engine()

    def keyPressEvent(self, event): # Keyboard input
        key = event.key()
        current_widget = self.stacked_widget.currentWidget()

        if current_widget == self.main_menu_screen:
            if key == Qt.Key_Escape: self.close()
        elif self.is_game_active:
            if hasattr(current_widget, 'customKeyPressEvent'): 
                if current_widget.customKeyPressEvent(event): 
                    return 
            elif hasattr(current_widget, 'keyPressEvent'): # Allow widget to handle its own keys if not custom
                # This might lead to double processing if not careful.
                # Better to use customKeyPressEvent or ensure widget's keyPressEvent calls super() appropriately or not at all.
                # For now, we prioritize customKeyPressEvent.
                pass


            if key == Qt.Key_Escape:
                self.return_to_main_menu()
                return

            if self.active_game_type == "crossword":
                text = event.text().upper()
                if key == Qt.Key_R: self.crossword_engine.process_key_press("NEW")
                elif self.crossword_engine.game_over_message and text != "NEW" and key != Qt.Key_R: pass # Only allow NEW when game over
                elif key == Qt.Key_H: self.crossword_engine.process_key_press("HINT")
                elif key == Qt.Key_Backspace or key == Qt.Key_Delete: self.crossword_engine.process_key_press("DEL")
                elif key == Qt.Key_Return or key == Qt.Key_Enter: self.crossword_engine.process_key_press("ENTER")
                elif key == Qt.Key_N: self.crossword_engine.process_key_press("NEXT")
                elif text.isalpha() and len(text)==1: self.crossword_engine.process_key_press(text)
                elif key in [Qt.Key_Right, Qt.Key_Down]: self.crossword_engine.advance_cursor_in_word()
                elif key in [Qt.Key_Left, Qt.Key_Up] and self.crossword_engine.current_selected_cell_idx_in_entry > 0:
                    self.crossword_engine.current_selected_cell_idx_in_entry -= 1
                else: super().keyPressEvent(event); return
                self._update_crossword_ui_from_engine()

            # Gesture and BrainBlinker handle their keys internally via their own keyPressEvent methods,
            # or customKeyPressEvent if defined. No specific block needed here unless for global keys
            # not caught by the widgets.
        else:
            super().keyPressEvent(event)


    def closeEvent(self, event):
        print("UI: Closing application...")
        self.is_game_active = False 

        if self.camera_thread and self.camera_thread.isRunning():
            print("UI: Stopping shared CameraThread on app close.")
            self.camera_thread.stop()
            if not self.camera_thread.wait(1500):
                print("UI: Shared CameraThread timeout during app close. Terminating.")
                self.camera_thread.terminate()
                self.camera_thread.wait() 
            else:
                print("UI: Shared CameraThread stopped on app close.")
        
        if hasattr(self, 'brain_blinker_game_screen') and hasattr(self.brain_blinker_game_screen, 'stop_game_or_pause'):
             print("UI: Ensuring BrainBlinker game is stopped on app close.")
             self.brain_blinker_game_screen.stop_game_or_pause()
        
        if hasattr(self, 'gesture_game_widget_instance') and hasattr(self.gesture_game_widget_instance, 'stop_game_or_pause'):
            print("UI: Ensuring HandGesture game is stopped on app close.")
            self.gesture_game_widget_instance.stop_game_or_pause()

        print("UI: Application cleanup finished.")
        event.accept()

# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())