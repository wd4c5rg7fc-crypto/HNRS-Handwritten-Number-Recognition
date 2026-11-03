import cv2
from operator_recognition import recognise_operator

image = cv2.imread("test_images/gui_segments/component_4.png")

print("Operator prediction:", recognise_operator(image))