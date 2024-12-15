import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                #We will get the coordinates of the landmarks and each id of the landmark
                #We will get the coordinates as decimal values, which are actually ratios. To get the pixel value, we need to multiply it with the screen width or height.
                h,w,c=img.shape
                cx , cy=int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv.circle(img, (cx, cy), 7, (255,0,0), 5, cv.FILLED)
                    
        return self.lmList
    
    def findFingersUp(self):
        # ##FOR RIGHT HAND ONLY##
        fingers = []
        # if tip is below the middle point of the finger, then we are considering that the finger is folded

        # Thumb
        if self.lmList[4][1] < self.lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for i in range(8,21,4):
            if self.lmList[i][2] < self.lmList[i-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    pTime = 0
    cTime = 0

    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        
        lmList = detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)

        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()