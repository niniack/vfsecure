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
    key = '0101'
    hash = "beefe1dddd1f8ecc33e5b8f0b0a0ae737eff02a71b39c4c5ef4ebae8b794089b"
    print(len(hash))
    ResizeMTX = None
    fogdim = None
    coords = None
    rotates = None
    dt = None

    @classmethod
    def genKey(cls):
        key = random.sample(range(9),4)
        key = key[0]*1000 + key[1]*100 + key[2]*10 + key[3]
        print(key)

    @classmethod
    def readMatrix(cls):
        # read image
        img = imread('../images/barcode.png')
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
        cls.ResizeMTX = []

        # reading data matrix image
        while row < hgt:
            while col < hgt:
                value = img[row,col]
                Resizerow.append(value)
                col = col + cellsize
                col1 = col1 + 1

            cls.ResizeMTX.append(Resizerow)
            Resizerow = []
            row = row + cellsize
            row1 = row1 + 1
            col1 = 0
            col = 0

    @classmethod
    def CODExy(cls):
        cls.ResizeMTX

        a = np.array(cls.ResizeMTX)
        codedim = a.shape[1]
        factor = 2#input("Fog Factor:")
        cls.fogdim = math.ceil(factor*codedim)

        offset = math.sqrt(2) * codedim/2
        x = 0
        y = 0

        while ((x-(cls.fogdim/2))**2)+((y-cls.fogdim/2)**2) > math.sqrt((cls.fogdim/2)**2-offset):
            x = random.randint(0,cls.fogdim)
            y = random.randint(0,cls.fogdim)

        buff = 1

        cls.dt = []
        for xi in range(int(x-codedim/2)-buff,int(x+codedim-codedim/2)+buff):
            for yi in range(int(y-codedim/2)-buff,int(y+codedim-codedim/2)+buff):
                XYi = [xi,yi]
                cls.dt.append(XYi)

        leftbuff = x - codedim/2
        rightbuff = cls.fogdim - leftbuff
        axis = 0

        pad_size = rightbuff - a.shape[axis]
        axis_nb = len(a.shape)
        if pad_size >= 0:
            npad = [(0, 0) for x in range(axis_nb)]
            npad[axis] = (int(leftbuff), int(pad_size))
            a = np.pad(a, pad_width=npad, mode='constant', constant_values=0)

        leftbuff = y - codedim/2
        rightbuff = cls.fogdim - leftbuff
        axis = 1

        pad_size = rightbuff - a.shape[axis]
        axis_nb = len(a.shape)
        if pad_size >= 0:
            npad = [(0, 0) for x in range(axis_nb)]
            npad[axis] = (int(leftbuff), int(pad_size))
            b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)

        cls.codecoords = []
        for p in range(cls.fogdim):
            for o in range(cls.fogdim):
                if b[p][o]:
                    XYc = [p,o]
                    cls.codecoords.append(XYc)

    @classmethod
    def FOGxy(cls):
        cls.fogdim
        count = 0
        density = 0.5
        noc =  math.floor(density *((cls.fogdim/2)**2)*3.14)
        cls.coords = []
        redo = 1

        for i in range(noc):
            x = random.randint(0,cls.fogdim)
            y = random.randint(0,cls.fogdim)
            XY = [x,y]
            redo = 0
            if XY in cls.coords:
                count = count + 1
                redo = 1
            while ((x-cls.fogdim/2)**2) + ((y-cls.fogdim/2)**2) > (cls.fogdim/2)**2 or redo:
                x = random.randint(0,cls.fogdim)
                y = random.randint(0,cls.fogdim)
                XY = [x, y]
                redo = 0
                if XY in cls.coords:
                    count = count + 1
                    redo = 1

            cls.coords.append(XY)

        for c in range(len(cls.dt)):
            if cls.dt[c] in cls.coords:
                cls.coords.remove(cls.dt[c])

        cls.coords = cls.coords + cls.codecoords

    @classmethod
    def assignZ(cls):
        cls.coords
        count = 0

        for t in range(len(cls.coords)):
            x = cls.coords[t][0]
            y = cls.coords[t][1]
            z = random.randint(0,cls.fogdim)

            while ((x-cls.fogdim/2)**2) + ((y-cls.fogdim/2)**2) + ((z-cls.fogdim/2)**2) > (cls.fogdim/2)**2 and count < 10000:
                count = count + 1
                z = random.randint(0,cls.fogdim)
            count = 0
            cls.coords[t].append(z)

        def Sort(sub_li):
            sub_li.sort(key = lambda x: (x[0], x[1]))
            return sub_li

    @classmethod
    def displayresults(cls):
        xvals = []
        yvals = []
        zvals = []
        for i in range(len(cls.coords)):
            xvals.append(cls.coords[i][0])
            yvals.append(cls.coords[i][1])
            zvals.append(cls.coords[i][2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xvals, yvals, zvals, c='r', marker='o')
        plt.show()

    @classmethod
    def genRotVal(cls):

        def truncate(f, n):
            '''Truncates/pads a float f to n decimal places without rounding'''
            s = '{}'.format(f)
            if 'e' in s or 'E' in s:
                return '{0:.{1}f}'.format(f, n)
            i, p, d = s.partition('.')
            return '.'.join([i, (d+'0'*n)[:n]])

        cls.rotates = []
        for a in range(len(cls.coords)):
            rot1 = 180*random.random()
            rot2 = 180*random.random()
            rot1 = float(truncate(rot1, 5))
            rot2 = float(truncate(rot2, 5))
            cls.rotates.append([rot1, rot2])

        model = int(str(cls.key)[:2])

        multi = int(str(cls.key)[2:4])
        print(model,multi)

        mr1 = 90*random.random()
        mr2 = 90*random.random()
        mr1 = float(truncate(mr1, 5))
        mr2 = float(truncate(mr2, 5))
        print(mr1,mr2)

        posvec = []
        extra = 0
        for x in range(1,33):
            digits = cls.hash[2*(x-1):2*x]
            rot1 = (mr1 + (int(digits[0], 16) * 180/16 - random.random() * 180/16)) % 180
            rot2 = (mr2 + (int(digits[1], 16) * 180/16 - random.random() * 180/16)) % 180
            rot1 = float(truncate(rot1, 5))
            rot2 = float(truncate(rot2, 5))
            cell = ((multi)*x) - 1 - extra
            pos = cell % len(cls.coords)
            while pos in posvec:
                extra = extra + 1
                cell = ((multi)*x) - 1 + extra
                pos = cell % len(cls.coords)

            posvec.append(pos)
            cls.rotates[pos][0] = rot1
            cls.rotates[pos][1] = rot2



        if model in posvec:
            x = 33
            cell = ((multi)*x) - 1 + extra
            pos = cell % len(cls.coords)
            cls.rotates[pos] = cls.rotates[model-1]
            print('ALLLLLLL')

        cls.rotates[model-1] = [mr1,mr2]
        #rot1 is about x axis
        #rot2 is about z axis
        print(np.array(cls.rotates))
    @classmethod
    def readSTL(cls):
        model =  int(str(output)[:2])
        multi =  int(str(output)[2:4])

    @classmethod
    def readRot(cls):
        # read model offset from sphere
        amod = function(model,origin)


        # read all of the cells
        posvec = []
        extra = 0
        for x in range(1,33):
            cell = ((multi)*x) - 1 - extra
            pos = cell % len(cls.coords)
            while pos in posvec:
                extra = extra + 1
                cell = ((multi)*x) - 1 + extra
                pos = cell % len(cls.coords)
            unhash[x-1] = function(pos,amod)
            posvec.append(pos)


        if model in posvec:
            x = 33
            cell = ((multi)*x) - 1 + extra
            pos = cell % len(cls.coords)
            cls.rotates[pos][0] = cls.rotates[model-1][1]


    #@classmethod
    #def decypherRot(cls):


def main():
    mesh = generator()
    mesh.genKey()
    mesh.readMatrix()
    mesh.CODExy()
    mesh.FOGxy()
    mesh.assignZ()
    mesh.genRotVal()


    # co = np.array(mesh.coords)
    # print(co)
    #print(np.array(mesh.rotates))
    # print(ro)
    # with open('../sphere/Origins.txt','w+') as f:
    #     np.savetxt(f, co)
    #     f.close()
    # with open('../sphere/Angles.txt','w+') as ff:
    #     np.savetxt(ff, ro)
    #     ff.close()

if __name__ == '__main__':
    main()
