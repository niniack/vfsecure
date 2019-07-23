#!/usr/bir/env python3
# https://barcode.tec-it.com/en/DataMatrix?data=Hi
# generate a 10 by 10 code with no quiet space
#python3 ./dmgen.pv

from matplotlib.image import imread, imsave
import numpy as np
import operator
import math
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylibdmtx.pylibdmtx import encode

from pprintpp import pprint as pp

from utils import _b


class generator:

    @classmethod
    def readMatrix(cls):
        # read image
        img = imread('../images/barcode3.png')
        # takes data from one of the RGB channels
        img = img[:,:,0]
        # measure height of the matrix
        hgt = len(img)
        # image must be inverted to paint the black cells
        if not img[0,0]:
            img = np.logical_not(img)

        # initialize cellsize
        cellsize = 1
        # initializing col
        col = hgt-1 #minus one because 0 is 1

        # finds cellsize pixel by pixel
        while not img[0,col-1]:
            cellsize = cellsize + 1
            col = col - 1

        # initializing for indexing position of all black cells
        row = 0
        col = 0
        row1 = 0
        col1 = 0
        Resizerow = []
        ResizeMTX = []

        # reading data matrix image
        while row < hgt:
            while col < hgt:
                value = img[row,col]
                Resizerow.append(value)
                col = col + cellsize
                col1 = col1 + 1

            ResizeMTX.append(Resizerow)
            Resizerow = []
            row = row + cellsize
            row1 = row1 + 1
            col1 = 0
            col = 0


    @classmethod
    def generateFOG(cls):


def main():
    mesh = generator()
    mesh.readMatrix()


if __name__ == '__main__':
    main()
