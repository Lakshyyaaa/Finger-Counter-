import time
import HandTrackingFunctionalP as htm
import cv2 as cv
import os

path = 'Fingers'
mylist = os.listdir(path)
overlaylist = []
curr_time=0
prev_time=0
counter=0
lm=[]
tipIds=[4,8,12,16,20]

capture = cv.VideoCapture(0)

for im in mylist:
    fin = cv.imread(f'{path}/{im}')
    if fin is not None:
        overlaylist.append(fin)

det=htm.HandDetector()


while True:
    isTrue, frame = capture.read()
    frame=det.showPoints(frame)
    lm=det.returnPoints(frame)
    fingers = []

    if len(lm)!=0:
        # Thumb
        if (lm[tipIds[0]][1] < lm[tipIds[0] - 1][1]):
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            #Fingers
            if (lm[tipIds[id]][2] < lm[tipIds[id]-2][2]):
                fingers.append(1)
            else:
                fingers.append(0)

    total=fingers.count(1)
    print(total)


    curr_time=time.time()
    fps=1/(curr_time-prev_time)
    prev_time=curr_time

    cv.putText(frame,str(int(fps)),(1700,150),2,cv.FONT_HERSHEY_TRIPLEX,(0,255,0),thickness=2)
    frame[0:300, 0:300] = overlaylist[total-1]
    cv.putText(frame,str(int(total)),(1700,250),4,cv.FONT_HERSHEY_PLAIN,(0,0,0),thickness=2)
    cv.imshow('Finger Counter', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
