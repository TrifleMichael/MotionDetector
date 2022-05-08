import cv2
import numpy as np


# Modifying derived images modifies source images!
# Example: adding contours to prepared_frame *at the end*, adds contours to diff_frame, dilated_diff_frame etc.
def my_motion_detector():
    cap = cv2.VideoCapture('test2.mp4')
    previous_frame = None
    while True:

        # Basic frame conversion
        ret, frame = cap.read()
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(9, 9), sigmaX=0)

        # Adding pressing Escape to close window
        if cv2.waitKey(30) == 27:
            break

        # Calculating difference between frames
        if previous_frame is not None:
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)

            # Dilating difference frame to make differences more visible
            kernel = np.ones((7, 7))
            dilated_diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # Filtering differences through threshold
            thresh_frame = cv2.threshold(src=dilated_diff_frame, thresh=30, maxval=255, type=cv2.THRESH_BINARY)[1]

            # CONTOURS TEMPORARLY OFF BECAUSE THEY DONT WORK TOO GOOD
            # Finding and adding difference contours
            contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) < 250:
                    # too small: skip!
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

            cv2.imshow('Video', frame)


        previous_frame = prepared_frame



my_motion_detector()