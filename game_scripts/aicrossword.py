# aicrossword.py
import random
import time
# Remove unused QtWidget/QtGui imports if they are not used elsewhere in this file
# from PyQt5.QtWidgets import QLabel, QPushButton
# from PyQt5.QtCore import QTimer, QRect, Qt, QPropertyAnimation
# from PyQt5.QtGui import QColor, QPainter

from PyQt5.QtCore import QObject, pyqtSignal # <<< REQUIRED IMPORTS

# --- Configuration specific to crossword logic (can be adjusted or passed in) ---
WORD_CLUE_BANK_LOGIC = [
    ("APPLE", "A common fruit, often red or green"), ("DOG", "A pet that barks"), ("CAT", "A pet that meows"),
    ("SUN", "The star our planet orbits"), ("MOON", "Earth's natural satellite"), ("STAR", "Twinkles in the night sky"),
    ("BOOK", "Something you read for stories"), ("HOUSE", "Where people live"), ("TREE", "Has leaves and branches"),
    ("WATER", "You drink this to stay hydrated"), ("HAPPY", "Feeling good and cheerful"), ("SMILE", "You do this when happy"),
    ("EARTH", "The planet we live on"), ("OCEAN", "A very large body of salt water"), ("FISH", "Swims in water, has fins"),
    ("BIRD", "Has feathers and can fly"), ("FLOWER", "A colorful plant part"), ("GRASS", "Green ground cover"),
    ("SCHOOL", "Place where children learn"), ("PENCIL", "Used for writing or drawing"), ("PAPER", "You write on this"),
    ("GAME", "Something fun to play"), ("MUSIC", "Sounds made with instruments or voice"), ("DANCE", "Moving your body to music"),
    ("FRIEND", "Someone you like to spend time with"), ("FAMILY", "People you are related to"), ("BALL", "A round toy for games"),
    ("CLOUD", "White fluffy thing in the sky"), ("RAIN", "Water falling from clouds"), ("SNOW", "Frozen water falling in winter"),
    ("CAKE", "A sweet dessert for birthdays"), ("JUICE", "Drink made from fruit"), ("SLEEP", "You do this at night in bed")
]
DEFAULT_PUZZLE_FALLBACK_LOGIC = [
    {"word": "FUN", "clue": "Enjoyable activity", "row": 1, "col": 1, "direction": 0},
    {"word": "PLAY", "clue": "To engage in games", "row": 1, "col": 1, "direction": 1}
]
POINTS_PER_CORRECT_WORD_LOGIC = 10
HINT_PENALTY_LOGIC = -2
POINTS_PENALTY_REVEAL_LOGIC = -5


class CrosswordGame(QObject): # <<< MODIFIED: Inherit from QObject
    game_completed_signal = pyqtSignal(int) # <<< ADDED: Define the signal

    def __init__(self):
        super().__init__() # <<< ADDED: Call QObject constructor
        # --- Game State Variables ---
        self.PUZZLE = []
        self.GRID_ROWS, self.GRID_COLS = 10, 10
        self.CELL_SIZE = 38 # UI might override this based on display space
        self.solution_grid = []
        self.player_grid = []
        self.player_grid_cell_status = []
        self.active_cells = []
        self.clue_numbers_map = {}
        self.current_clue_idx = 0
        self.current_entry_cells = []
        self.current_selected_cell_idx_in_entry = 0
        self.SCORE = 0
        self.game_over_message = ""
        self.feedback_message_info = {"text": "", "is_good": True}
        self.solved_word_indices = set()

        self._word_bank = WORD_CLUE_BANK_LOGIC[:]
        self._default_puzzle = DEFAULT_PUZZLE_FALLBACK_LOGIC[:]
        self._reset_game_state() # Initialize state on creation


    def _reset_game_state(self):
        self.PUZZLE, self.solution_grid, self.player_grid, self.player_grid_cell_status = [], [], [], []
        self.active_cells, self.clue_numbers_map, self.current_entry_cells = [], {}, []
        self.GRID_ROWS, self.GRID_COLS = 10, 10 # Default, generate_crossword will update
        self.SCORE, self.current_clue_idx, self.current_selected_cell_idx_in_entry = 0, 0, 0
        self.game_over_message = ""; self.solved_word_indices = set()
        self.set_feedback("", True)
        print("Crossword logic: Game state reset.")

    def set_feedback(self, message, is_good=True):
        self.feedback_message_info = {"text": message, "is_good": is_good}

    def start_new_game(self):
        self._reset_game_state()
        print("Crossword logic: Starting new game generation...")
        if self.generate_crossword_puzzle(self._word_bank, random.randint(4, 7)):
            self.initialize_crossword()
        else:
            print("Crossword logic: Generation failed, using fallback.")
            self.PUZZLE = self._default_puzzle[:]
            # Recalculate grid dims for fallback if main generation failed
            p_max_r, p_max_c = 0, 0
            for p_item in self.PUZZLE:
                r, c, word, direction = p_item["row"], p_item["col"], p_item["word"], p_item["direction"]
                if direction == 0: p_max_r = max(p_max_r, r); p_max_c = max(p_max_c, c + len(word) - 1)
                else: p_max_r = max(p_max_r, r + len(word) - 1); p_max_c = max(p_max_c, c)
            self.GRID_ROWS, self.GRID_COLS = max(5, p_max_r + 2), max(5, p_max_c + 2)
            self.initialize_crossword()

    def generate_crossword_puzzle(self, word_bank, num_words_target):
        temp_puzzle_data = []
        if not word_bank: temp_puzzle_data = self._default_puzzle[:]
        else:
            num_select = min(num_words_target, len(word_bank), 8)
            if num_select <= 0: temp_puzzle_data = self._default_puzzle[:]
            else:
                selected = random.sample(word_bank, num_select)
                opts = [(1,1,0),(1,1,1),(1,4,0),(2,1,1),(2,5,0),(3,2,1),(3,6,0),(4,3,1),(4,1,0),(1,6,1),(5,2,0),(2,7,1)]
                random.shuffle(opts); opt_idx = 0
                for i, (word, clue) in enumerate(selected):
                    if opt_idx < len(opts): r, c, d = opts[opt_idx]; opt_idx += 1
                    else: r, c, d = (1 + (i % 4)*2, 1 + (i // 4)*2, i % 2) # Fallback placement
                    temp_puzzle_data.append({"word": word.upper(), "clue": clue, "row": r, "col": c, "direction": d})

        max_r, max_c = 0, 0
        for p in temp_puzzle_data:
            r_val, c_val, w_val, d_val = p["row"], p["col"], p["word"], p["direction"] # Renamed for clarity
            if d_val == 0: max_r, max_c = max(max_r, r_val), max(max_c, c_val + len(w_val) - 1)
            else: max_r, max_c = max(max_r, r_val + len(w_val) - 1), max(max_c, c_val)
        self.GRID_ROWS, self.GRID_COLS = max(5, max_r + 2), max(5, max_c + 2)

        temp_sol_grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        final_puzzle = []
        temp_puzzle_data.sort(key=lambda x: len(x['word']), reverse=True)
        for p_item_gen in temp_puzzle_data: # Renamed for clarity
            r_gen, c_gen, w_gen, d_gen = p_item_gen["row"], p_item_gen["col"], p_item_gen["word"], p_item_gen["direction"]
            can_place = True
            cells_to_occupy = []
            for i, char in enumerate(w_gen):
                cr, cc = (r_gen, c_gen + i) if d_gen == 0 else (r_gen + i, c_gen)
                if not (0 <= cr < self.GRID_ROWS and 0 <= cc < self.GRID_COLS): can_place = False; break
                existing = temp_sol_grid[cr][cc]
                if existing != '' and existing != char: can_place = False; break
                cells_to_occupy.append(((cr, cc), char))
            if can_place:
                final_puzzle.append(p_item_gen)
                for (cell_r, cell_c), char_fill in cells_to_occupy: temp_sol_grid[cell_r][cell_c] = char_fill
        self.PUZZLE = final_puzzle
        if not self.PUZZLE:
            print("Crossword logic: All words conflicted, using fallback for puzzle content.")
            self.PUZZLE = self._default_puzzle[:]
            p_max_r, p_max_c = 0, 0
            for p_item_fb in self.PUZZLE: # Renamed for clarity
                r_fb, c_fb, word_fb, dir_fb = p_item_fb["row"], p_item_fb["col"], p_item_fb["word"], p_item_fb["direction"]
                if dir_fb == 0: p_max_r = max(p_max_r, r_fb); p_max_c = max(p_max_c, c_fb + len(word_fb) - 1)
                else: p_max_r = max(p_max_r, r_fb + len(word_fb) - 1); p_max_c = max(p_max_c, c_fb)
            self.GRID_ROWS, self.GRID_COLS = max(5, p_max_r + 2), max(5, p_max_c + 2)

        print(f"Crossword logic: Generated {len(self.PUZZLE)} words for grid {self.GRID_ROWS}x{self.GRID_COLS}")
        return bool(self.PUZZLE)

    def initialize_crossword(self):
        self.solution_grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.player_grid = [[' ' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.player_grid_cell_status = [['normal' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.active_cells = [[False for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.clue_numbers_map = {}; placed_starts = {}; clue_num_counter = 1

        if not self.PUZZLE: print("Error: Init with empty PUZZLE"); return

        for i, p_item in enumerate(self.PUZZLE):
            r, c, word, direction = p_item["row"], p_item["col"], p_item["word"], p_item["direction"]
            if (r, c) not in placed_starts:
                 num_str = str(clue_num_counter); placed_starts[(r, c)] = num_str
                 self.clue_numbers_map[(r, c)] = num_str; p_item["clue_num"] = clue_num_counter
                 clue_num_counter += 1
            else: p_item["clue_num"] = int(placed_starts[(r,c)]) # Ensure it's an int if reusing
            for char_idx, char_val in enumerate(word):
                curr_r, curr_c = (r, c + char_idx) if direction == 0 else (r + char_idx, c)
                if 0 <= curr_r < self.GRID_ROWS and 0 <= curr_c < self.GRID_COLS:
                    if self.solution_grid[curr_r][curr_c] != '' and self.solution_grid[curr_r][curr_c] != char_val:
                        print(f"LOGIC CONFLICT (should be rare): At ({curr_r},{curr_c}), existing '{self.solution_grid[curr_r][curr_c]}' vs '{char_val}' for '{word}'.")
                    self.solution_grid[curr_r][curr_c] = char_val; self.active_cells[curr_r][curr_c] = True
        self.PUZZLE.sort(key=lambda p: p.get("clue_num", float('inf'))) # Ensure clue_num exists or handle
        self.current_clue_idx = 0
        if self.PUZZLE: self.set_current_word_focus(0)
        print("Crossword logic: Crossword initialized.")

    def set_current_word_focus(self, clue_idx_val):
        if not self.PUZZLE: return
        self.current_clue_idx = clue_idx_val % len(self.PUZZLE)
        p_item = self.PUZZLE[self.current_clue_idx]
        self.current_entry_cells = []
        r, c, word, direction = p_item["row"], p_item["col"], p_item["word"], p_item["direction"]
        for i in range(len(word)):
            cell_r, cell_c = (r, c + i) if direction == 0 else (r + i, c)
            if 0 <= cell_r < self.GRID_ROWS and 0 <= cell_c < self.GRID_COLS:
                 self.current_entry_cells.append((cell_r, cell_c))
        self.current_selected_cell_idx_in_entry = 0
        if not self.current_entry_cells: return
        for idx, (cr, cc) in enumerate(self.current_entry_cells):
             if 0 <= cr < self.GRID_ROWS and 0 <= cc < self.GRID_COLS:
                 if self.player_grid_cell_status[cr][cc] == 'normal' or self.player_grid[cr][cc] == ' ':
                      self.current_selected_cell_idx_in_entry = idx; break

    def get_current_selected_cell(self):
        if self.current_entry_cells and 0 <= self.current_selected_cell_idx_in_entry < len(self.current_entry_cells):
            return self.current_entry_cells[self.current_selected_cell_idx_in_entry]
        return None

    def _check_word(self, clue_idx, check_player_grid=True):
        if not self.PUZZLE or not (0 <= clue_idx < len(self.PUZZLE)): return False, []
        p = self.PUZZLE[clue_idx]; r, c, w, d = p["row"], p["col"], p["word"], p["direction"]
        cells = []; is_match = True
        for i in range(len(w)):
            cr, cc = (r, c + i) if d == 0 else (r + i, c)
            if not (0 <= cr < self.GRID_ROWS and 0 <= cc < self.GRID_COLS): return False, [] # Word goes out of bounds
            cells.append((cr, cc))
            if check_player_grid and self.player_grid[cr][cc].upper() != self.solution_grid[cr][cc].upper():
                is_match = False
                if check_player_grid: break # No need to check further if one char is wrong
        return is_match, cells

    def check_and_score_word(self, clue_idx):
        if clue_idx in self.solved_word_indices: return False
        is_correct, word_cells = self._check_word(clue_idx, check_player_grid=True)
        if is_correct:
            self.SCORE += POINTS_PER_CORRECT_WORD_LOGIC
            self.set_feedback(f"Correct! +{POINTS_PER_CORRECT_WORD_LOGIC}", True)
            for sr, sc in word_cells:
                if self.player_grid_cell_status[sr][sc] not in ['hinted', 'revealed']:
                     self.player_grid_cell_status[sr][sc] = 'correct'
            self.solved_word_indices.add(clue_idx)
            return True
        return False

    def reveal_word(self, clue_idx):
        _, word_cells = self._check_word(clue_idx, check_player_grid=False) # Get cells for the solution
        if not word_cells: return
        penalty = 0
        if clue_idx not in self.solved_word_indices:
            penalty = POINTS_PENALTY_REVEAL_LOGIC
            self.SCORE += penalty
        for r_cell, c_cell in word_cells:
             self.player_grid[r_cell][c_cell] = self.solution_grid[r_cell][c_cell] # Fill from solution
             self.player_grid_cell_status[r_cell][c_cell] = 'revealed'
        word_str = self.PUZZLE[clue_idx]['word']
        self.set_feedback(f"Revealed: {word_str}. ({penalty} pts)", False)
        self.solved_word_indices.add(clue_idx)


    def process_key_press(self, key_char):
        if key_char == "NEW": self.start_new_game(); return
        if self.game_over_message: self.set_feedback("Game Over! Press 'NEW GAME' or 'R' to restart.", True); return
        if not self.PUZZLE: self.set_feedback("No puzzle loaded.", False); return

        idx = self.current_clue_idx
        selected_cell_coords = self.get_current_selected_cell()
        r_sel, c_sel = selected_cell_coords if selected_cell_coords else (-1, -1)
        in_bounds = 0 <= r_sel < self.GRID_ROWS and 0 <= c_sel < self.GRID_COLS

        if key_char == "HINT":
            if selected_cell_coords and idx not in self.solved_word_indices and in_bounds:
                is_editable = self.player_grid_cell_status[r_sel][c_sel] not in ['correct', 'revealed']
                is_wrong_or_empty = self.player_grid[r_sel][c_sel] == ' ' or \
                                    self.player_grid[r_sel][c_sel].upper() != self.solution_grid[r_sel][c_sel].upper()
                if is_editable and is_wrong_or_empty:
                    self.player_grid[r_sel][c_sel] = self.solution_grid[r_sel][c_sel].upper()
                    self.player_grid_cell_status[r_sel][c_sel] = 'hinted'
                    self.SCORE += HINT_PENALTY_LOGIC
                    self.set_feedback(f"Hint Used! ({HINT_PENALTY_LOGIC} pts)", False)
                    if self.check_word_state(idx): # Check if this hint completed the word
                        if self.check_and_score_word(idx): self.move_to_next_unsolved_clue()
                    else: self.advance_cursor_in_word() # If not complete, just move cursor
                else: self.set_feedback("Cell already correct!", True); self.advance_cursor_in_word()
            elif idx in self.solved_word_indices: self.set_feedback("Word already solved!", True); self.move_to_next_unsolved_clue()
            else: self.set_feedback("Cannot use hint here.", False)


        elif key_char == "DEL":
            if selected_cell_coords and in_bounds and self.player_grid_cell_status[r_sel][c_sel] not in ['correct', 'revealed', 'hinted']:
                 current_char_is_space = (self.player_grid[r_sel][c_sel] == ' ')
                 self.player_grid[r_sel][c_sel] = ' '
                 if current_char_is_space and self.current_selected_cell_idx_in_entry > 0:
                     self.current_selected_cell_idx_in_entry -= 1
                     prev_cell = self.get_current_selected_cell()
                     if prev_cell:
                         pr, pc = prev_cell
                         if 0 <= pr < self.GRID_ROWS and 0 <= pc < self.GRID_COLS and \
                            self.player_grid_cell_status[pr][pc] not in ['correct', 'revealed', 'hinted']:
                            self.player_grid[pr][pc] = ' '
                 elif not current_char_is_space: # If char was deleted (was not space), try to move back
                     if self.current_selected_cell_idx_in_entry > 0:
                        self.current_selected_cell_idx_in_entry -=1

            elif self.current_selected_cell_idx_in_entry > 0: # Cell locked, just move back
                self.current_selected_cell_idx_in_entry -= 1


        elif key_char == "ENTER":
            if idx in self.solved_word_indices: self.set_feedback("Word already solved!", True); self.move_to_next_unsolved_clue()
            elif self.check_and_score_word(idx): self.move_to_next_unsolved_clue()
            else: self.set_feedback("That's not quite right. Try again!", False)

        elif key_char == "NEXT": self.move_to_next_unsolved_clue()

        elif len(key_char) == 1 and key_char.isalpha():
            if selected_cell_coords and in_bounds:
                 if self.player_grid_cell_status[r_sel][c_sel] not in ['correct', 'revealed', 'hinted']:
                     self.player_grid[r_sel][c_sel] = key_char.upper()
                     self.player_grid_cell_status[r_sel][c_sel] = 'normal' # Mark as normal after input
                     self.advance_cursor_in_word()
                     # Check if the whole word is filled and correct
                     if self.check_word_state(idx): # Checks if current state of word is correct
                         if self.check_and_score_word(idx): # If correct, scores it and adds to solved
                             self.move_to_next_unsolved_clue()
                 else:
                     self.set_feedback("Can't change this letter!", False)
                     self.advance_cursor_in_word() # Still advance if possible even if uneditable
            else: # No selected cell, or out of bounds (should be rare if logic is correct)
                self.set_feedback("Select a cell first.", False)


    def advance_cursor_in_word(self):
         if not self.current_entry_cells: return
         start_idx = self.current_selected_cell_idx_in_entry
         # Try to move to the next cell in the current word entry
         if self.current_selected_cell_idx_in_entry < len(self.current_entry_cells) - 1:
             self.current_selected_cell_idx_in_entry += 1
         else: # At the end of the word, maybe check if filled or move to next clue's start
            # Current behavior: stays at the end. UI might trigger "ENTER" or word check.
            pass


    def check_word_state(self, clue_idx):
         # Checks if the currently entered player characters for the given word match the solution
         # This is used for auto-check after typing or after a hint.
         is_match, _ = self._check_word(clue_idx, check_player_grid=True)
         return is_match

    def move_to_next_unsolved_clue(self):
        if not self.PUZZLE or len(self.solved_word_indices) == len(self.PUZZLE):
            self.check_game_completion(); return # Should emit signal if game over
        
        start_idx = self.current_clue_idx
        for i in range(len(self.PUZZLE)): # Iterate at most once through all puzzles
            next_idx = (start_idx + 1 + i) % len(self.PUZZLE) # Ensure we check all from next one
            if next_idx not in self.solved_word_indices:
                self.set_current_word_focus(next_idx)
                return
        # If loop finishes, it means all are solved (or only current was unsolved and now is solved)
        self.check_game_completion() # Call again to ensure game over is processed if reached here


    # <<< THIS IS THE CORRECT check_game_completion TO MODIFY AND KEEP >>>
    def check_game_completion(self):
        if self.PUZZLE and not self.game_over_message and len(self.solved_word_indices) == len(self.PUZZLE):
            self.game_over_message = f"Awesome! All Solved! Score: {self.SCORE}"
            self.set_feedback(self.game_over_message, True)
            print("Crossword logic: Game Over! Emitting signal.") # For debugging
            self.game_completed_signal.emit(self.SCORE)  # <<< EMIT THE SIGNAL HERE
            return True
        return False

