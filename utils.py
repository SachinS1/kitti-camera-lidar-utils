import numpy as np
import cv2


def view_image(img_path):

    img = cv2.imread(img_path)

    cv2.imshow("Frame", img)

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def view_image_with_labels(img_path, label_path):
        img = cv2.imread(img_path)

        with open(label_path, "r") as file:
             lines = file.readlines()

        for line in lines:
            data = line.split()
            label = data[0] 
            x1, y1, x2, y2 = map(float, data[4:8])

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", img)

        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

def generate_bev_image(bin_file_path, configs):
     pass