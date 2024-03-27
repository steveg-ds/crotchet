import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(2)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
prev_wrist_pos = None

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get wrist position
            wrist_pos = (int(handLms.landmark[0].x * img.shape[1]), int(handLms.landmark[0].y * img.shape[0]))

            # If this is not the first frame and wrist position has changed, print the new position
            if prev_wrist_pos is not None and wrist_pos != prev_wrist_pos:
                print("Wrist position:", wrist_pos)

            # Update previous wrist position
            prev_wrist_pos = wrist_pos

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
