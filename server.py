import cv2
import mediapipe as mp
import time
from threading import Thread


# Open video capture device (e.g., webcam)
cap = cv2.VideoCapture(2)

# Initialize Mediapipe hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

pTime = 0
prev_wrist_pos = None
stitch_counter = 0
paused = False

# Function to reset paused state after a delay
def reset_paused_state():
    global paused
    time.sleep(5)  # 5 seconds pause
    paused = False

reset_paused_thread = Thread(target=reset_paused_state)
reset_paused_thread.start()

while True:
    # Read frame from the video capture device
    success, img = cap.read()
    if not success:
        break

    # Convert frame to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(imgRGB)

    # Draw landmarks and track wrist position
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get wrist position
            wrist_pos = (int(handLms.landmark[0].x * img.shape[1]), int(handLms.landmark[0].y * img.shape[0]))
            th = (int(handLms.landmark[4].x * img.shape[1]), int(handLms.landmark[4].y * img.shape[0]))
            mf = (int(handLms.landmark[12].x * img.shape[1]), int(handLms.landmark[12].y * img.shape[0]))

            if (img.shape[1] // 2) - 38 <= wrist_pos[0] <= (img.shape[1] // 2) + 38 and \
                    (img.shape[0] // 2) - 38 <= wrist_pos[1] <= (img.shape[0] // 2) + 38 and not paused:
                stitch_counter += 1
                print("Stitch Counter:", stitch_counter)
                paused = True
                reset_paused_thread = Thread(target=reset_paused_state)
                reset_paused_thread.start()

            if (5 * img.shape[1] // 6) - 30 <= mf[0] <= (5 * img.shape[1] // 6) + 30 and \
                    (img.shape[0] // 6) - 30 <= mf[1] <= (img.shape[0] // 6) + 30 and \
                    (5 * img.shape[1] // 6) - 30 <= th[0] <= (5 * img.shape[1] // 6) + 30 and \
                    (img.shape[0] // 6) - 30 <= th[1] <= (img.shape[0] // 6) + 30 and \
                    stitch_counter != 0 and not paused:
                stitch_counter = 0
                print("Stitch Counter:", stitch_counter)

    cv2.putText(img, f'Stitches: {stitch_counter}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 0, 90),
                2)

    center = (img.shape[1] // 2, img.shape[0] // 2)
    cv2.circle(img, center, 5, (255, 0, 0), -1)

    snap_dot = ((5 * img.shape[1]) // 6, img.shape[0] // 6)
    cv2.circle(img, snap_dot, 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Hand Tracking', img)

    # Check for the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture device and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
