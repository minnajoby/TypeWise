import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5
)

video = cv2.VideoCapture(0)
tipids = [4, 8, 12, 16, 20]
previous_positions = {}
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
                for tipid in tipids:
                    cx = lmlst[tipid][1]
                    cy = lmlst[tipid][2]
                    if tipid in previous_positions:
                        px,py=previous_positions[tipid]
                         # Calculate movement distance
                        distance = ((cx - px)**2 + (cy - py)**2)**0.5
                        cv2.putText(
                            img,
                            str(tipid) + str(round(distance, 4)),
                            (20, 40 + tipid * 3),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.6,
                            (0, 255, 0),
                            1
                        )
                    previous_positions[tipid]=(cx,cy)
            mp_draw.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    else:
        previous_positions.clear()
    cv2.imshow("TypeWise - Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()