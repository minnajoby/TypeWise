import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5
)

video = cv2.VideoCapture(0)
previous_x = None
previous_y = None
while True:
    suc,img = video.read()
    conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(conv_img)
    lmlst = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                cx = lm.x
                cy = lm.y
                lmlst.append([id, cx, cy])
            if len(lmlst) == 21:
                    # Index finger tip = landmark 8
                    cx = lmlst[8][1]
                    cy = lmlst[8][2]
                    # First frame: no previous position
                    if previous_x is not None and previous_y is not None:
                         # Calculate movement distance
                        distance = ((cx - previous_x)**2 + (cy - previous_y)**2)**0.5
                        cv2.putText(
                            img,
                            'Movement: ' + str(round(distance, 4)),
                            (30, 400),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (0, 255, 0),
                            1
                        )
                    # Store current position for next frame
                    previous_x = cx
                    previous_y = cy
            mp_draw.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    cv2.imshow("TypeWise - Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()