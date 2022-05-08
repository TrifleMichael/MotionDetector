import cv2
import numpy as np
import argparse


def detect_motion(path: str, mask_path: str, kernel_val: int, ksize: int, debug: bool, minobjsize: int):
    cap = cv2.VideoCapture(path)
    previous_frame = None

    mask = None
    if mask_path is not None:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply blur to remove of noise
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(ksize, ksize), sigmaX=0)

        # Adding pressing Escape to close window
        if cv2.waitKey(30) == 27:
            cv2.destroyAllWindows()
            break

        # Calculating difference between frames
        if previous_frame is not None:
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)

            # Dilating difference frame to make differences more visible
            kernel = np.ones((kernel_val, kernel_val))
            dilated_diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # Filtering differences through threshold
            thresh_frame = cv2.threshold(
                src=dilated_diff_frame,
                thresh=30,
                maxval=255,
                type=cv2.THRESH_BINARY
            )[1]

            # Applying mask
            if mask is not None:
                h, w = thresh_frame.shape
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                thresh_frame = cv2.multiply(thresh_frame, mask)

            # Finding contours
            contours, _ = cv2.findContours(
                image=thresh_frame,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Process only contours, which are big enough
                if cv2.contourArea(contour) < minobjsize:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

            cv2.imshow('Video', frame)
            if debug:
                cv2.imshow('Diff', diff_frame)
                cv2.imshow('Threshold', thresh_frame)

        previous_frame = prepared_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("Source", type=str, nargs="?")
    parser.add_argument("--mask", type=str, nargs="?", const="", default=None)
    parser.add_argument("--debug", type=bool, nargs="?", const=True, default=False)
    parser.add_argument("--ksize", type=int, nargs="?", const=9, default=9)
    parser.add_argument("--kernel", type=int, nargs="?", const=7, default=7)
    parser.add_argument("--minobjsize", type=int, nargs="?", const=250, default=250)
    args = parser.parse_args()
    video_path = args.Source
    mask_path = args.mask
    debug = args.debug
    ksize = args.ksize
    kernel_val = args.kernel
    detect_motion(video_path or 0, mask_path, kernel_val, ksize, debug, args.minobjsize)
