import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5
)

video = cv2.VideoCapture(0)
while True:
    suc,img = video.read()
    conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(conv_img)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    cv2.imshow("TypeWise - Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()