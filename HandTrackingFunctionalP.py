import cv2 as cv
import mediapipe as mp
import time

class HandDetector:
    mpHands = mp.solutions.hands
    Hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    def showPoints(self, frame):
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.Hands.process(img)
        if results.multi_hand_landmarks:
            for handLMS in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, handLMS, self.mpHands.HAND_CONNECTIONS)
        return frame

    def returnPoints(self, frame):
        points_list = []
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.Hands.process(img)
        if results.multi_hand_landmarks:
            h, w, c = frame.shape
            for handLMS in results.multi_hand_landmarks:
                for id, lm in enumerate(handLMS.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points_list.append([id, cx, cy])
        return points_list


def main():
    capture = cv.VideoCapture(0)
    curr_time = 0
    prev_time = 0
    detector = HandDetector()

    while True:
        isTrue, frame = capture.read()

        frame = detector.showPoints(frame)
        points = detector.returnPoints(frame)

        if points:
            print(points[4])

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv.putText(frame, f"FPS: {int(fps)}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        f=cv.flip(frame,1)
        cv.imshow('Hand Tracking', f)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
