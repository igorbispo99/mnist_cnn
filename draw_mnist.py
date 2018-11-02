import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import cv2 as cv
import numpy as np

from keras.models import load_model
from scipy.misc import imresize


CURRENT_FRAME = np.zeros((560, 560))
DRAW_MODE = False

def mouse_event(event, x, y, flags, params):
    global CURRENT_FRAME, DRAW_MODE

    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_LBUTTONUP:
        DRAW_MODE = not(DRAW_MODE)

    elif event == cv.EVENT_MOUSEMOVE:
        if DRAW_MODE:
            cv.circle(CURRENT_FRAME, (x, y), 10, 255, -1)

def process_image():
    return np.array(1/255 * imresize(CURRENT_FRAME, (28, 28))).reshape((1, 28, 28, 1))

def main():
    global CURRENT_FRAME

    model = load_model("mnist_cnn.h5")

    window_name = "Frame"
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_event) 

    while(1):
        cv.imshow(window_name, CURRENT_FRAME)
        key = cv.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('p'):
            pred = model.predict_on_batch(process_image())
            print("Predicted => {}".format(pred.argmax()))
        elif key == ord('c'):
            CURRENT_FRAME = np.zeros((560, 560))


    cv.destroyAllWindows()


if __name__ == '__main__':
    main()