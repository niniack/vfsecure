#!/usr/bir/env python3
from matplotlib.image import imread, imsave
import numpy as np
import operator
img = imread('barcode.png')

img = img[:,:,0]
hgt = len(img)
if not img[0,0]:
    img = np.logical_not(img)
cellsize = 0
col = - 1 #minus one because 0 is 1
while not img[0,col-1]:
    cellsize = cellsize + 1
    col = col - 1
row = 0
col = 0
row1 = 0
col1 = 0
Resizerow = []
ResizeMTX = []
while row <= hgt:
    while col <= hgt:
        value = img[row,col]
        Resizerow = [Resizerow,value]

        col = col + cellsize
        col1 = col1 + 1

    ResizeMTX = [ResizeMTX,Resizerow]
    row = row + cellsize
    row1 = row1 + 1
    col1 = 0
    col = 0
