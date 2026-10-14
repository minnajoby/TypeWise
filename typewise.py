import cv2
import mediapipe as mp
import time
from pynput import keyboard

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5
)

video = cv2.VideoCapture(0)
tipids = [4, 8, 12, 16, 20]
previous_positions = {}
total_movement = {
    4: 0,
    8: 0,
    12: 0,
    16: 0,
    20: 0
}
fingernames = {
    4: "Thumb",
    8: "Index",
    12: "Middle",
    16: "Ring",
    20: "Pinky"
}
key_count = 0
start_time = time.time()

def on_press(key):
    global key_count

    # Count typing keys, not modifier keys
    if hasattr(key, 'char') and key.char is not None:
        key_count += 1

    elif key == keyboard.Key.space:
        key_count += 1

listener = keyboard.Listener(on_press=on_press)
listener.start()

while True:
    suc,img = video.read()
    conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(conv_img)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lmlst = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx = lm.x
                cy = lm.y
                lmlst.append([id, cx, cy])
            if len(lmlst) == 21:
                for tipid in tipids:
                    cx = lmlst[tipid][1]
                    cy = lmlst[tipid][2]
                    if tipid in previous_positions:
                        px,py=previous_positions[tipid]
                        # Calculate movement distance
                        distance = ((cx - px)**2 + (cy - py)**2)**0.5
                        total_movement[tipid] += distance
                        cv2.putText(
                            img,
                            fingernames[tipid] + ': ' +
                            str(round(total_movement[tipid], 3)),
                            (20, 40 + tipids.index(tipid) * 35),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.6,
                            (0, 255, 0),
                            1
                        )
                    previous_positions[tipid]=(cx,cy)
            mp_draw.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    else:
        previous_positions.clear()
    elapsed_time = time.time() - start_time

    cv2.putText(
        img,
        'Key Presses: ' + str(key_count),
        (20, 230),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (255, 0, 0),
        2
    )
    cv2.putText(
        img,
        'Time: ' + str(round(elapsed_time, 1)) + ' sec',
        (20, 265),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (255, 0, 0),
        2
    )
    cv2.imshow("TypeWise - Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
listener.stop()
video.release()
cv2.destroyAllWindows()