#!/usr/bir/env python3
# Materials Used: https://barcode.tec-it.com/en/DataMatrix?data=Hi
# Description: generate a 10 by 10 code with no quiet space
# Authors: Michael Linares (@michaellinares) and Nishant Aswani (@niniack)


from matplotlib.image import imread, imsave
import numpy as np
import operator
import math
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from pylibdmtx.pylibdmtx import encode

from pprintpp import pprint as pp

from utils import _b


class generator:

    #dmgen vars
    key = '0205'
    hash = "beefe1dddd1f8ecc33e5b8f0b0a0ae737eff02a71b39c4c5ef4ebae8b794089b"
    # print(len(hash))
    ResizeMTX = None
    fogdim = None
    origins = None
    angles = None
    dt = None

    #genspheres vars
    faces=None
    vertices=None
    normals=None
    shiftedVertices=None
    numFaces=None
    numVertices=None

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
        cls.origins = []
        redo = 1

        for i in range(noc):
            x = random.randint(0,cls.fogdim)
            y = random.randint(0,cls.fogdim)
            XY = [x,y]
            redo = 0
            if XY in cls.origins:
                count = count + 1
                redo = 1
            while ((x-cls.fogdim/2)**2) + ((y-cls.fogdim/2)**2) > (cls.fogdim/2)**2 or redo:
                x = random.randint(0,cls.fogdim)
                y = random.randint(0,cls.fogdim)
                XY = [x, y]
                redo = 0
                if XY in cls.origins:
                    count = count + 1
                    redo = 1

            cls.origins.append(XY)

        for c in range(len(cls.dt)):
            if cls.dt[c] in cls.origins:
                cls.origins.remove(cls.dt[c])

        cls.origins = cls.origins + cls.codecoords

    @classmethod
    def assignZ(cls):
        # cls.origins
        count = 0

        for t in range(len(cls.origins)):
            x = cls.origins[t][0]
            y = cls.origins[t][1]
            z = random.randint(0,cls.fogdim)

            while ((x-cls.fogdim/2)**2) + ((y-cls.fogdim/2)**2) + ((z-cls.fogdim/2)**2) > (cls.fogdim/2)**2 and count < 10000:
                count = count + 1
                z = random.randint(0,cls.fogdim)
            count = 0
            cls.origins[t].append(z)

        def Sort(sub_li):
            sub_li.sort(key = lambda x: (x[0], x[1]))
            return sub_li

    @classmethod
    def displayresults(cls):
        xvals = []
        yvals = []
        zvals = []
        for i in range(len(cls.origins)):
            xvals.append(cls.origins[i][0])
            yvals.append(cls.origins[i][1])
            zvals.append(cls.origins[i][2])

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

        cls.angles = []
        for a in range(len(cls.origins)):
            rot1 = 90*random.random()
            rot2 = 90*random.random()
            rot1 = float(truncate(rot1, 5))
            rot2 = float(truncate(rot2, 5))
            cls.angles.append([rot1, rot2])

        model = int(str(cls.key)[:2])
        multi = int(str(cls.key)[2:4])

        mr1 = 90*random.random()
        mr2 = 90*random.random()
        mr1 = float(truncate(mr1, 5))
        mr2 = float(truncate(mr2, 5))

        posvec = []
        extra = 0
        for x in range(1,33):
            digits = cls.hash[2*(x-1):2*x]
            rot1 = (mr1 + (int(digits[0], 16) * 90/16 - random.random() * 90/16)) % 90
            rot2 = (mr2 + (int(digits[1], 16) * 90/16 - random.random() * 90/16)) % 90
            rot1 = float(truncate(rot1, 5))
            rot2 = float(truncate(rot2, 5))
            cell = ((multi)*x) - 1 - extra
            pos = cell % len(cls.origins)
            while pos in posvec:
                extra = extra + 1
                cell = ((multi)*x) - 1 + extra
                pos = cell % len(cls.origins)

            posvec.append(pos)
            cls.angles[pos][0] = rot1
            cls.angles[pos][1] = rot2
            # print(x)

        if model in posvec:
            x = 33
            cell = ((multi)*x) - 1 + extra
            pos = cell % len(cls.origins)
            # print(cls.angles[pos])
            cls.angles[pos] = cls.angles[model-1]
            # print(cls.angles[pos])

        cls.angles[model-1][0] = mr1
        cls.angles[model-1][1] = mr2

    @classmethod
    def readModel(cls):
        # open two file buffers to read faces and vertices data
        vd = open('../sphere/vertices.txt','rb')
        fd = open('../sphere/faces.txt', 'rb')

        # load the buffers into arrays
        cls.vertices = np.loadtxt(vd,np.float32)
        cls.faces = np.loadtxt(fd,np.int32)

        # close the buffers to avoid data being overwritten or corrupted
        vd.close()
        fd.close()

        # convert from MATLAB indexing to Python indexing by subtracting 1
        cls.faces = np.subtract(cls.faces,1)

        cls.numFaces = np.size(cls.faces,0)
        cls.numVertices = np.size(cls.vertices,0)

    @classmethod
    def shift(cls, index, rotateBool):

        shifted = cls.vertices + cls.origins[index]

        numFaces = np.size(cls.faces,0)

        if (rotateBool == True):
            shifted = cls.rotate(cls.origins[index], shifted, index)

        for i in range(numFaces):

            v0 = int(cls.faces[i][0])
            v1 = int(cls.faces[i][1])
            v2 = int(cls.faces[i][2])

            vec1 = shifted[v1] - shifted[v0]
            vec2 = shifted[v2] - shifted[v0]

            cls.normals[(numFaces*index)+i]=np.cross(vec1, vec2)

        for i in range(cls.numVertices):
            cls.shiftedVertices[(cls.numVertices*index)+i]=shifted[i]

    @classmethod
    def rotate(cls, origin, vertices, index):

        # append a column of 1s to the vertices for matrix transformations
        rawVertices = np.hstack((vertices, np.ones([np.size(vertices, 0),1])))

        # Transformation matrix 1 to move vertex by diff between sphere origin and 0,0,0
        t1 = np.identity(4)
        t1[3,:] = np.hstack((-origin,1))

        ######################## ROTATION #########################3
        rollAngle = np.deg2rad(cls.angles[index][0])
        roll = np.array([[1, 0, 0, 0], [0, np.cos(rollAngle), np.sin(rollAngle), 0], [0, -np.sin(rollAngle), np.cos(rollAngle), 0], [0, 0, 0, 1]] )

        yawAngle = np.deg2rad(cls.angles[index][1])
        yaw = np.array([[np.cos(yawAngle), np.sin(yawAngle), 0, 0], [-np.sin(yawAngle), np.cos(yawAngle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ######################## ROTATION #########################


        # Transformatio matrix 2 to move vertex back to original location
        t2 = np.identity(4)
        t2[3,:] = np.hstack((origin,1))

        rawVertices = np.matmul(rawVertices, t1)
        rawVertices = np.matmul(rawVertices, roll)
        rawVertices = np.matmul(rawVertices, yaw)
        rawVertices = np.matmul(rawVertices, t2)


        rawVertices = np.delete(rawVertices, 3, 1)
        # pp(rawVertices)
        return rawVertices

    @classmethod
    def writeSTL(cls):

        numSpheres = np.size(cls.origins,0)

        #normalizing normals
        cls.normals =  cls.normals*(1/np.sqrt(np.add(cls.normals*cls.normals,1)))

        wf = open('sphere.stl', 'wb+')
        wf.write(_b("\0"*80))

        wf.write(_b(np.uint32(cls.numFaces*numSpheres)))

        for i in range(cls.numFaces*numSpheres):
            wf.write(_b(np.float32(cls.normals[i][0])))
            wf.write(_b(np.float32(cls.normals[i][1])))
            wf.write(_b(np.float32(cls.normals[i][2])))
            for j in range(3):

                sphereNum = math.floor(i/cls.numFaces)
                indicesFromSphereZero = cls.numFaces*sphereNum

                wf.write(_b(np.float32(cls.shiftedVertices[ cls.faces[i-indicesFromSphereZero][j]+(cls.numVertices*sphereNum) ][0])))
                wf.write(_b(np.float32(cls.shiftedVertices[ cls.faces[i-indicesFromSphereZero][j]+(cls.numVertices*sphereNum) ][1])))
                wf.write(_b(np.float32(cls.shiftedVertices[ cls.faces[i-indicesFromSphereZero][j]+(cls.numVertices*sphereNum) ][2])))

            wf.write(_b(np.uint16(0)))

        wf.close()

def main():
    # create mesh object
    mesh = generator()
    # reading matrix
    mesh.readMatrix()
    # x,y coordinates for matrix cells
    generator.CODExy()
    # x,y coordinates for FOG
    mesh.FOGxy()
    # z coordinates for each x,y
    mesh.assignZ()

    # print(len(mesh.origins))

    # encode hash in rotation
    mesh.genRotVal()
    # read model sphere from matlab data
    mesh.readModel()

    # store number of spheres
    generator.numSpheres = len(mesh.origins)

    # convert origins and angles from python list to np array
    generator.origins = np.array(generator.origins)
    generator.angles = np.array(generator.angles)

    # pp(mesh.origins)

    # initialize array for normals
    generator.normals = np.zeros([mesh.numFaces*mesh.numSpheres,3])

    # initialize array for shiftedVertices
    generator.shiftedVertices = np.zeros([mesh.numVertices*mesh.numSpheres,3])

    # shift each sphere by its origin value
    # The boolean controls rotation: True = rotate; False = no rotate;
    for i in range(mesh.numSpheres): #numSpheres
        mesh.shift(i, True)

    # write out STL file
    mesh.writeSTL()

    # FOR DEBUGGING DECODER PURPOSES
    np.savetxt("../legacy/angles.txt", mesh.angles)


if __name__ == '__main__':
    main()
