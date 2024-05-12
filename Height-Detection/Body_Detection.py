import cv2 as cv
import mediapipe as mp
import time
import numpy as np

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils

pose = mpPose.Pose()
capture = cv.VideoCapture(0)
lst = []
n = 0
ptime = 0

while True:
    isTrue, img = capture.read()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            lst.append([id, lm.x, lm.y])
            n += 1
            h, w, c = img.shape
            if id == 11:  # Left shoulder landmark
                cx1, cy1 = int(lm.x * w), int(lm.y * h)
                cv.circle(img, (cx1, cy1), 15, (0, 0, 0), cv.FILLED)
            if id == 23:  # Left hip landmark
                cx2, cy2 = int(lm.x * w), int(lm.y * h)
                cy2 = cy2 + 20
                cv.circle(img, (cx2, cy2), 15, (0, 0, 0), cv.FILLED)
                # Calculate height
                d = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
                di = round(d * 0.5)
                print(f"You are {di} centimeters tall")
                print("I am done")
                print("You can relax now")
                print("Press q and give me some rest now.")
                if ord('q'):
                    cv.destroyAllWindows()
    img = cv.resize(img, (1280, 720))
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(img, "FPS : ", (40, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), thickness=2)
    cv.putText(img, str(int(fps)), (160, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), thickness=2)
    cv.imshow("Task", img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()