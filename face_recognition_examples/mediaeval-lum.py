# import the necessary packages

from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

movframes = "/home/yt/Desktop/cvpr2014/repro/mediaeval/data/dataset/movframes/"
filename = movframes + 'Sintel.mp4-00012.jpg'

def colorhist1(filename):
    # draw histogram in python.

    img = cv2.imread(filename)
    h = np.zeros((300, 256, 3))

    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([img], [ch], None, [256], [0, 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)

    h = np.flipud(h)

    cv2.imshow('colorhist', h)
    cv2.waitKey(0)

def colorhist3(filename):
    # read original image, in full color, based on command
    # line argument
    img = cv2.imread(filename)

    # display the image
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)

    # split into channels
    channels = cv2.split(img)

    # list to select colors of each channel line
    colors = ("b", "g", "r")

    # create the histogram plot, with three lines, one for
    # each color
    plt.figure()
    plt.xlim([0, 256])
    for (channel, c) in zip(channels, colors):
        histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(histogram, color=c)

    plt.xlabel("Color value")
    plt.ylabel("Pixels")

    plt.show()

def hsvhist(filename):
    image = cv2.imread(filename)
    #cv2.imshow("Original Image", image)
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    #plt.figure()
    #plt.title("'Flattened' Color Histogram")
    #plt.xlabel("Bins")
    #plt.ylabel("# of Pixels")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist,1, cv2.NORM_MINMAX)
        features.extend(hist)

        # plot the histogram
        #plt.plot(hist, color=color)
        #plt.xlim([0, 256])

    #plt.show()
    #print("flattened feature vector size: %d" % (np.array(features).flatten().shape))
    return np.array(features).flatten()

def colorhist(filename):
    image = cv2.imread(filename)
    #cv2.imshow("Original Image", image)
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    #plt.figure()
    #plt.title("'Flattened' Color Histogram")
    #plt.xlabel("Bins")
    #plt.ylabel("# of Pixels")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist,1, cv2.NORM_MINMAX)
        features.extend(hist)

        # plot the histogram
        #plt.plot(hist, color=color)
        #plt.xlim([0, 256])

    #plt.show()
    #print("flattened feature vector size: %d" % (np.array(features).flatten().shape))
    return np.array(features).flatten()



def grayhist(filename):
    image = cv2.imread(filename)
    #cv2.imshow("image", image)
    #convert the image to grayscale and create a histogram
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("gray", gray)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    #cv2.normalize(hist, hist,0,1, cv2.NORM_MINMAX)
    cv2.normalize(hist, hist, 1, cv2.NORM_MINMAX)

    #print(type( hist ))
    #print(hist)
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

    return np.array(hist).flatten()




#colorhist3(filename)
#colorhist1(filename)
print(grayhist(filename))
#print(colorhist(filename))




while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27: break             # ESC key to exit
cv2.destroyAllWindows()

