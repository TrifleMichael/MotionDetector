import cv2
import numpy as np
import argparse


def detect_motion(path: str, mask_path: str, kernel_val: int, ksize: int, debug: bool):
    cap = cv2.VideoCapture(path)
    previous_frame = None

    mask = None
    if mask_path is not None:
        print(mask_path)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    while True:
        # Basic frame conversion
        ret, frame = cap.read()
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO parameterize this with command line
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(ksize, ksize), sigmaX=0)

        # Adding pressing Escape to close window
        if cv2.waitKey(30) == 27:
            break

        # Calculating difference between frames
        if previous_frame is not None:
            # TODO [*]
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)

            # Dilating difference frame to make differences more visible
            # TODO parameterize this with command line
            kernel = np.ones((kernel_val, kernel_val))
            dilated_diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # Filtering differences through threshold
            thresh_frame = cv2.threshold(src=dilated_diff_frame, thresh=30, maxval=255, type=cv2.THRESH_BINARY)[1]

            # Applying mask
            if mask is not None:
                h, w = thresh_frame.shape
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                thresh_frame = cv2.multiply(thresh_frame, mask)

            # TODO [*]
            # THRESH_FRAME

            # Finding and adding difference contours
            contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) < 250:
                    # too small: skip!
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

            # TODO pokazywanie ramki like this
            cv2.imshow('Video', frame)
            if debug:
                cv2.imshow('Diff', diff_frame)


        previous_frame = prepared_frame

# python .\main.py [source] [--mask maska.png]
# brak source oznacza kamerkę z komputera
# python .\main.py test2.mp4 --mask mask_rightupper.png

if __name__ == '__main__':
    # TODO add --debug mode which displays all the frames marked with [*]
    parser = argparse.ArgumentParser()
    parser.add_argument("Source", type=str, nargs="?", const="")
    parser.add_argument("--mask", type=str, nargs=1)
    parser.add_argument("--debug", type=bool, nargs="?", const=False, default=False)
    parser.add_argument("--ksize", type=int, nargs="?", const=9, default=9)
    parser.add_argument("--kernel", type=int, nargs="?", const=7, default=7)
    args = parser.parse_args()
    video_path = args.Source
    mask_path = args.mask
    debug = args.debug
    ksize = args.ksize
    kernel_val = args.kernel
    print(kernel_val)
    detect_motion(video_path or 0, mask_path[0], kernel_val, ksize, debug)
