import cv2 as cv
import numpy as np
import os
import HandTrackingModule as htm
import tkinter as tk
from tkinter import messagebox
import math

######## VARIABLES ########

brushThickness = 15
eraserThickness = 25

brushThicknessMin = 1
brushThicknessMax = 50

###########################

# close confirmation window
def show_confirmation():
    root = tk.Tk()
    root.withdraw()
    if messagebox.askokcancel("Close", "Do you really want to close?"):
        cap.release()
        cv.destroyAllWindows()
    root.destroy()

current_dir = os.path.dirname(os.path.abspath(__file__))
folderPath = os.path.join(current_dir, "assets")

uiFolder = "NEW UI 2"

myList = os.listdir(os.path.join(folderPath, uiFolder))

overlayList = []

for imPath in myList:
    
    image_path = os.path.join(folderPath, uiFolder, imPath)
    
    image = cv.imread(image_path)
    
    if image is not None:
        overlayList.append(image)

header = overlayList[5]
sidebar = cv.imread(os.path.join(folderPath, "sidebar.png"))
helpBox = cv.imread(os.path.join(folderPath, "helpbg6.png"))
colour = (0,0,0)

cap = cv.VideoCapture(1)
cap.set(3, 640) # DO NOT CHANGE
cap.set(4, 480) # DO NOT CHANGE

detector = htm.handDetector(detectionCon=0.85,trackCon=0.5)

xp, yp = 0, 0

imgCanvas = np.zeros((480,640,3), dtype='uint8') # DO NOT CHANGE

while True:
    # importing image
    success, img = cap.read()
    img = cv.flip(img, 1)
    
    # finding landmarks
    img = detector.findHands(img, draw = True)
    lmList = detector.findPosition(img, draw = False)

    # lm of tip on index and middle
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x4, y4 = lmList[20][1:]

        # checking which fingers are up
        fingers = detector.findFingersUp()
        # print(fingers)

        # selection mode - two fingers 
        if fingers[1] and fingers[2] and fingers[0]==False and fingers[3]==False and fingers[4]==False:
            # checking click
            if (y1 < 70):
                if 350<x1<398:
                    header = overlayList[4]
                    colour = (80,83,239)
                elif 409<x1<457:
                    header = overlayList[3]
                    colour = (218,198,38)
                elif 468<x1<515:
                    header = overlayList[2]
                    colour = (207,227,73)
                elif 527<x1<574:
                    header = overlayList[1]
                    colour = (255,255,255)
                elif 586<x1<633:
                    header = overlayList[0]
                    colour = (0,0,0)


            #closing
            if x1<50 and 196<y1<229:
                show_confirmation()

            #help
            if x1<50 and 141<y1<175:
                img[70:,50:] = helpBox

            #clear
            if x1<50 and 86<y1<120:
                imgCanvas = np.zeros((480,640,3), dtype='uint8')

            if colour == (0,0,0):
                cv.circle(img, (x1,y1), eraserThickness, colour)
            else:
                cv.circle(img, (x1,y1), brushThickness, colour)

            xp, yp = 0, 0
        
        if fingers[1]==False:
            xp, yp = 0, 0

        # drawing mode - index finger
        if fingers[1] and fingers[2]==False and fingers[0]==False and fingers[3]==False and fingers[4]==False:

            if xp==0 and yp==0:
                xp, yp = x1, y1

            if colour == (0,0,0):
                cv.circle(img, (x1,y1), eraserThickness, colour, cv.FILLED)
                cv.line(img, (xp,yp), (x1,y1), color = colour, thickness=eraserThickness)
                cv.line(imgCanvas, (xp,yp), (x1,y1), color = colour, thickness=eraserThickness)
            else:
                cv.circle(img, (x1,y1), brushThickness, colour, cv.FILLED)
                cv.line(img, (xp,yp), (x1,y1), color = colour, thickness=brushThickness)
                cv.line(imgCanvas, (xp,yp), (x1,y1), color = colour, thickness=brushThickness)

            xp, yp = x1, y1

        # size change mode - index and little finger
        if fingers[1] and fingers[4] and fingers[0]==False and fingers[2]==False and fingers[3]==False:

            cv.circle(img, (x1,y1), 15, colour, cv.FILLED)
            cv.circle(img, (x4,y4), 15, colour, cv.FILLED)
            cv.line(img, (x1,y1),(x4,y4),colour, 5)
            c1, c2 = (x1+x4)//2, (y1+y4)//2
            d1, d2 = c1 + 50, c2 - 10

            if colour != (0,0,0):
                cv.circle(img, (c1,c2), brushThickness, colour, cv.FILLED)
                cv.putText(img, str(brushThickness), (d1, d2), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

                length = math.hypot(x4-x1,y4-y1)
                brushThickness = int(np.interp(length, [5,200], [brushThicknessMin,brushThicknessMax]))

            else:
                cv.circle(img, (c1,c2), eraserThickness, colour, cv.FILLED)
                cv.putText(img, str(eraserThickness), (d1, d2), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

                length = math.hypot(x4-x1,y4-y1)
                eraserThickness = int(np.interp(length, [5,200], [brushThicknessMin,brushThicknessMax]))


    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)

    #setting the header and sidebar
    img[:70,:] = header
    img[70:,:50] = sidebar

    cv.imshow("Image", img)

    cv.waitKey(1)