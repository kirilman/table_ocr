from main import Table
import cv2
import numpy as np


def test_Table():
    table = Table()
    img = cv2.imread("../good/230913050_page_0.jpg")
    assert type(img, np.array)
