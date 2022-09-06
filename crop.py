import cv2
from imutils import contours as C


def crop(file, min_t, max_t, xb, yb):
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    r, t = cv2.threshold(gray, min_t, max_t, cv2.THRESH_BINARY_INV)
    contours, h = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    (contours, _) = C.sort_contours(contours)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 1000:
            break

    cropped_img = image[y - yb:y + h + yb, x - xb:x + w + xb]
    # cv2.imshow('cropped', cropped_img)
    # cv2.waitKey(0)
    return cropped_img
