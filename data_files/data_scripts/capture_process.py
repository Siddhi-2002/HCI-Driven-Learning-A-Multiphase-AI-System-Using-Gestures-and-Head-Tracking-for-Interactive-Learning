import cv2
import os
import time
import numpy as np
import math
from PIL import Image
from cvzone.HandTrackingModule import HandDetector

# ---- 1. Get class folder from user ----
folder_name = input("Enter the folder name (0–5): ").strip()

if not folder_name.isdigit() or not (0 <= int(folder_name) <= 5):
    print("Invalid input! Enter a number between 0 and 5.")
    exit()

# Paths for original and processed data
original_path = os.path.join(
    "C:\\Users\\Siddhi Mohanty\\Desktop\\hand gesture detection game\\DeepLearn_Hand_Gesture_Recognition_for_Early_Learning\\data_files\\data_sets\\data_original",
    folder_name  
)
grayscale_path = os.path.join(
    "C:\\Users\\Siddhi Mohanty\\Desktop\\hand gesture detection game\\DeepLearn_Hand_Gesture_Recognition_for_Early_Learning\\data_files\\data_sets\\data_grayscale_resized",
    folder_name  
)

os.makedirs(original_path, exist_ok=True)
os.makedirs(grayscale_path, exist_ok=True)

# ---- 2. Setup image counter ----
existing_files = [f for f in os.listdir(original_path) if f.endswith('.jpg')]
count = len(existing_files)

# ---- 3. Initialize camera and hand detector ----
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
last_capture_time = time.time()

# ---- 4. Loop to capture and save hand images ----
while True:
    success, img = cap.read()
    if not success:
        print("Camera error.")
        break

    # Flip the image for mirror preview
    flipped_img = cv2.flip(img, 1)

    # Detect hands on the original image
    hands, img = detector.findHands(img)

    if hands:
        hand_crops = []

        for hand in hands:
            x, y, w, h = hand['bbox']
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]
            if imgCrop.size == 0:
                continue

            # Flip cropped image
            imgCrop = cv2.flip(imgCrop, 1)

            aspectRatio = h / w
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            if aspectRatio > 1:
                k = imgSize / h
                wCal = min(imgSize, math.ceil(k * w))
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - imgResize.shape[1]) // 2
                imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
            else:
                k = imgSize / w
                hCal = min(imgSize, math.ceil(k * h))
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - imgResize.shape[0]) // 2
                imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

            hand_crops.append(imgWhite)

        # Combine if 2 hands
        if len(hand_crops) == 2:
            combined = np.hstack((hand_crops[0], hand_crops[1]))
            combined = cv2.resize(combined, (imgSize * 2, imgSize))
        else:
            combined = hand_crops[0]

        # Save images once per second
        current_time = time.time()
        if current_time - last_capture_time >= 1:
            filename = f"{count}.jpg"
            original_file = os.path.join(original_path, filename)
            grayscale_file = os.path.join(grayscale_path, filename)

            # Save original
            cv2.imwrite(original_file, combined)

            # Convert to grayscale and resize
            try:
                img_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
                img_gray = img_pil.convert('L')
                img_resized = img_gray.resize((128, 128))
                img_resized.save(grayscale_file)
                print(f"Saved: {original_file} & {grayscale_file}")
            except Exception as e:
                print(f"Error saving grayscale image: {e}")

            count += 1
            last_capture_time = current_time

        # Show output
        cv2.imshow("Processed Hand(s)", combined)

    # Show mirror preview
    cv2.putText(flipped_img, f"Preview | Class: {folder_name} | Count: {count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Mirror Preview - Press Q to quit", flipped_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- 5. Cleanup ----
cap.release()
cv2.destroyAllWindows()
print("Capture session ended.")
