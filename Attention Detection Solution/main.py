import cv2
import dlib
import random
import datetime
import keyboard
import time
import csv
import pandas as pd
from playsound import playsound
import matplotlib.pyplot as plt

with open('attention_time_file.cvs', 'w') as fa:
    writer = csv.writer(fa)
    writer.writerow(["Time[min]"])

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
tStart = datetime.datetime.now()

entryLoop1 = 1
entryLoop2 = 1
startCount = 0.0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    isItFace = len(faces)
    voiceNum = random.randint(1, 7)

    if isItFace == 0 and entryLoop1 == 1:
        entryLoop1 = 0
        startCount = time.time()
        print('ne prati')

    if isItFace == 1:
        startCount = time.time()

    for face in faces:

        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        landmarks = predictor(gray, face)
        x33 = landmarks.part(33).x
        x01 = landmarks.part(1).x
        x15 = landmarks.part(15).x
        x02 = landmarks.part(2).x
        x14 = landmarks.part(14).x
        x03 = landmarks.part(3).x
        x13 = landmarks.part(13).x
        y37 = landmarks.part(37).y
        y38 = landmarks.part(38).y
        y40 = landmarks.part(40).y
        y41 = landmarks.part(41).y
        y43 = landmarks.part(43).y
        y44 = landmarks.part(44).y
        y46 = landmarks.part(46).y
        y47 = landmarks.part(47).y

        if (x33 == x01) or (x33 == x03) or (x33 == x02) or \
                (x33 == x15) or (x33 == x14) or (x33 == x13) or \
                (y37 == y41) or (y38 == y40) or \
                (y43 == y47) or (y44 == y46):

            br=voiceNum.__str__()

            playsound(r"C:\Users\Aleks\Desktop\Attention Detection Solution\GlasoviWAv\glas"+br+".wav")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            timeEnd = datetime.datetime.now()
            delta_time = timeEnd - tStart
            sec = delta_time.seconds
            sec=sec/60

            with open('attention_time_file.cvs', 'a') as fa:
                writer = csv.writer(fa)
                writer.writerow([int(sec)])
        print(face)

        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow("Frame 1", frame)

    if time.time() - startCount > 3:
        br=voiceNum.__str__()
        playsound(r"C:\Users\Aleks\Desktop\Attention Detection Solution\GlasoviWAv\glas" +br+".wav")

        timeEnd = datetime.datetime.now()
        delta_time = timeEnd - tStart
        entryLoop1 = 1
        sec = delta_time.seconds
        sec = sec/60

        with open('attention_time_file.cvs', 'a') as fa:
            writer = csv.writer(fa)
            writer.writerow([int(sec)])

    key = cv2.waitKey(1)
    if key == 27:
        break
    if keyboard.is_pressed('esc'):
        break

df = pd.read_csv('attention_time_file.cvs', usecols=["Time[min]"])

secon = df["Time[min]"]
range = (0, 45)
bins = 15

plt.hist(secon, bins, range, color='green',
         histtype='bar', rwidth=0.8)

plt.xlabel('Time in minutes')
plt.ylabel('No. of no attention')
plt.title('Attention histogram')
plt.show()
