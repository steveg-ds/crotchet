from flask import Flask, Response
import cv2
import mediapipe as mp
import time
from threading import Thread

# Initialize Flask application
app = Flask(__name__)


# Function to generate video frames with hand tracking
def generate_frames():
    # Open video capture device (e.g., webcam)
    cap = cv2.VideoCapture(2)

    # Initialize Mediapipe hands module
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    stitch_counter = 0
    paused = False

    # Function to reset paused state after a delay
    def reset_paused_state():
        nonlocal paused
        time.sleep(5.5)  # 5 seconds pause
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
                wrist_pos = (int(handLms.landmark[8].x * img.shape[1]), int(handLms.landmark[8].y * img.shape[0]))
                th = (int(handLms.landmark[4].x * img.shape[1]), int(handLms.landmark[4].y * img.shape[0]))
                mf = (int(handLms.landmark[12].x * img.shape[1]), int(handLms.landmark[12].y * img.shape[0]))
                # print(mf)
                if (img.shape[1] // 2) - 30 <= wrist_pos[0] <= (img.shape[1] // 2) + 30 and \
                        (img.shape[0] // 2) - 30 <= wrist_pos[1] <= (img.shape[0] // 2) + 30 and not paused:
                    stitch_counter += 1
                    print("Stitch Counter:", stitch_counter)
                    paused = True
                    reset_paused_thread = Thread(target=reset_paused_state)
                    reset_paused_thread.start()

                if 100 - 40 <= mf[0] <= 100 + 40 and \
                        1000 - 40 <= mf[1] <= 1000 + 40 and \
                        stitch_counter != 0 and not paused:
                    stitch_counter = 0
                    print("Stitch Counter:", stitch_counter)

        # Display stitch counter on the image
        img = cv2.flip(img, -1)
        cv2.putText(img, f'Stitches: {stitch_counter}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                    3)

        cv2.putText(img, f'save me', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                    3)

        center = (img.shape[1] // 2, img.shape[0] // 2)
        cv2.circle(img, center, 10, (255, 255, 255), -1)

        snap_dot = (1000,100)
        cv2.circle(img, snap_dot, 10, (0, 0, 255), -1)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()

        # Yield frame bytes to be streamed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release video capture device
    cap.release()


# Route for streaming video with hand tracking
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Main function to run the application
if __name__ == '__main__':
    app.run(host='192.168.1.174', port=60000, debug=True, use_reloader=True)

