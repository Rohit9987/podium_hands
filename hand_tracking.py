"""
Google - MediaPipe Framework

HandTracking
2 modules 
    -palm detection
    -hand landmarks

21 different landmarks
    0 - wrist
    1,4 - thumb
    5-8 - index
    9-12 - middle
    13-16 - ring
    17-20 - little
"""

import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    #print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 8:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)
                    #print(id, cx, cy)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
            (255, 0, 255), 3)

    cv.imshow("Image", img)
    cv.waitKey(1)


