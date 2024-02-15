import time

import cv2 as cv
import os

camera = cv.VideoCapture(0)
if not camera.isOpened():
    print("The Camera is not Opened....Exiting")
    exit()

Labels = ["None"]
for label in Labels:
    if not os.path.exists(label):
        os.mkdir(label)

for folder in Labels:
    count = 0
    print("Press 's' to start data collection for "+folder)
    userinput = input()
    if userinput != 's':
        print("Wrong Input..........")
        exit()
    status, frame = camera.read()
    cv.imshow("Video Window", frame)
    time.sleep(3)
    while count<1000:
        status, frame = camera.read()
        cv.imshow("Video Window",frame)
        if not status:
            print("Frame is not been captured..Exiting...")
            break
        cv.imwrite('./'+folder+"./"+str(count)+'.jpg',frame)
        count=count+1
        if cv.waitKey(1) == ord('q'):
            break
camera.release()
cv.destroyAllWindows()