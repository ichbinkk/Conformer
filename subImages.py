import cv2
import os


def subIm(image1, image2, i):
    # # load images
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    # compute difference
    difference = cv2.subtract(image1, image2)

    # color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference[mask != 255] = [255, 255, 255]

    # add the red mask to the images to make the differences obvious
    image1[mask != 255] = [0, 0, 255]
    image2[mask != 255] = [0, 0, 255]

    # store images
    # cv2.imwrite('diffOverImage1.png', image1)
    # cv2.imwrite('diffOverImage2.png', image1)
    cv2.imwrite('./output/' + str(i) + '.png', difference)

    # cv2.imshow("Subtracted",difference)
    # cv2.waitKey()

if __name__ == '__main__':
    t = 350
    dir = '../8-11-2'
    for i in range(t):
        image1 = os.path.join(dir, str(i)+'.png')
        image2 = os.path.join(dir, str(i+1) + '.png')
        subIm(image1, image2,i+1)