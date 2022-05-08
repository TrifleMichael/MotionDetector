import cv2
import numpy as np


# Modifying derived images modifies source images!
# Example: adding contours to prepared_frame *at the end*, adds contours to diff_frame, dilated_diff_frame etc.
def my_motion_detector():
    cap = cv2.VideoCapture('testVid2.mp4')
    previous_frame = None
    while True:

        # Basic frame conversion
        ret, frame = cap.read()
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

        # Adding pressing Escape to close window
        if cv2.waitKey(30) == 27:
            break

        # Calculating difference between frames
        if previous_frame is not None:
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)

            # Dilating difference frame to make differences more visible
            kernel = np.ones((5, 5))
            dilated_diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # Filtering differences through threshold
            thresh_frame = cv2.threshold(src=dilated_diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

            # CONTOURS TEMPORARLY OFF BECAUSE THEY DONT WORK TOO GOOD
            # Finding and adding difference contours
            # contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            # for contour in contours:
            #     if cv2.contourArea(contour) < 50:
            #         # too small: skip!
            #         continue
            #     (x, y, w, h) = cv2.boundingRect(contour)
            #     cv2.rectangle(img=prepared_frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

            cv2.imshow('Video', diff_frame)


        previous_frame = prepared_frame



my_motion_detector()


def motion_detector():
    frame_count = 0
    previous_frame = None

    while True:
        frame_count += 1

        # 1. Load image; convert to RGB
        img_brg = np.array(ImageGrab.grab())
        img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

        if ((frame_count % 2) == 0):
            # 2. Prepare image; grayscale and blur
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

        # 3. Set previous frame and continue if there is None
        if (previous_frame is None):
            # First frame; there is no previous one yet
            previous_frame = prepared_frame
            continue

        # calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
        previous_frame = prepared_frame

        # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        # 5. Only take different areas that are different enough (>20 / 255)
        thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)

        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                # too small: skip!
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

        #cv2.imshow('Motion detector', frame_rgb)

        if (cv2.waitKey(30) == 27):
            break