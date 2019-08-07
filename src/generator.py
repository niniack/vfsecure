#!/usr/bir/env python3
# Materials Used: https://barcode.tec-it.com/en/DataMatrix?data=Hi
# Description: generate a 10 by 10 code with no quiet space
# Authors: Michael Linares (@michaellinares) and Nishant Aswani (@niniack)


from matplotlib.image import imread, imsave
import argparse
import numpy as np
import operator
from utils import _b
import math
import random
import hashlib
import stl
from stl import mesh
from mpl_toolkits import mplot3d
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from pylibdmtx.pylibdmtx import encode
from PIL import Image
import hashlib
import os.path
import sys

from pprintpp import pprint as pp

from utils import _b
from utils import _standardWriteSTL

HEADER_COUNT = 80


class generator:

    #dmgen vars
    key = None
    hash = None
    # print(len(hash))
    ResizeMTX = None
    fogdim = None
    origins = None
    angles = None
    dt = None
    img = None

    #genspheres vars
    faces=None
    vertices=None
    normals=None
    shiftedVertices=None
    numFaces=None
    numVertices=None

    #insert vars
    code = None
    part = None
    combined = None
    newOrigin = [0,0,0]
    dispData = [0,0,0]
    offset = [96,69]

    @classmethod
    def genKey(cls):
        cls.key = random.randint(0,9999)

        cls.key = "%04d" % cls.key

    @classmethod
    def shufflePart(cls, filename):

        # Create data object to store values
        cls.dtObj = np.dtype([
                ('normals', np.float32, (3,)),
                ('vertices', np.float32, (3, 3)),
                ('attrs', np.uint16, (1,))
                ])

        rf = open(filename, 'r', encoding='ascii', errors='replace')


        # Store data in numpy arrays
        header = np.fromfile(rf, dtype=np.uint8, count=HEADER_COUNT)
        numTri = int(np.fromfile(rf, dtype=np.uint32, count=1))
        part = np.fromfile(rf, dtype=cls.dtObj, count=-1)

        # Close read buffer to avoid corruption
        rf.close()

        # Shuffle array of vertices/faces
        np.random.shuffle(part)

        filename = '../stl/shuffledPart.stl'

        # Write shuffled STL file of part (from utils)
        _standardWriteSTL(filename=filename, numTri=numTri, normals=part['normals'], vertices=part['vertices'])

    @classmethod
    def genMatrix(cls):
        encoded = encode(cls.key.encode('utf8'), scheme='Ascii', size='10x10')
        cls.img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels)

        cls.img.save('../images/DMTX.png')

    @classmethod
    def readMatrix(cls):
        # read image
        img = imread('../images/DMTX.png')
        # takes data from one of the RGB channels
        img = img[:,:,0]
        img = np.array(img)
        # measure height of the matrix
        hgt = len(img)
        # image must be inverted to paint the black cells
        if img[0,0]:
            img = np.logical_not(img)
        # initializing col
        col = hgt-1 #minus one because 0 is 1

        # counting whitespace
        middle = round(hgt/2)
        r=0
        while not img[middle][r]:
            r=r+1

        ws=r
        cs=0
        while img[ws][r]:
            r=r+1
            cs=cs+1

        # delete whitespace
        for d in range(ws):
            print(0)
            end = len(img)-1
            img = np.delete(img, end, 1)
            img = np.delete(img, 0, 1)
            img = np.delete(img, end, 0)
            img = np.delete(img, 0, 0)

        # initializing for indexing position of all black cells
        row = 0
        col = 0
        hgt = len(img)
        Resizerow = []
        cls.ResizeMTX = []

        # reading data matrix image
        while row < hgt:
            while col < hgt:
                Resizerow.append(img[row,col])
                col = col + cs

            cls.ResizeMTX.append(Resizerow)
            Resizerow = []
            row = row + cs
            col = 0

    @classmethod
    def genHash(cls):
        cls.hash = hashlib.sha256(b'../stl/shuffledPart.stl').hexdigest()


    @classmethod
    def CODExy(cls):
        cls.ResizeMTX

        a = np.array(cls.ResizeMTX)
        codedim = a.shape[1]
        factor = 1.42#input("Fog Factor:")
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
    def genRotVal(cls):

        def truncate(f, n):
            '''Truncates/pads a float f to n decimal places without rounding'''
            s = '{}'.format(f)
            if 'e' in s or 'E' in s:
                return '{0:.{1}f}'.format(f, n)
            i, p, d = s.partition('.')
            return '.'.join([i, (d+'0'*n)[:n]])


        spec = 90
        cls.angles = []
        for a in range(len(cls.origins)):
            rot1 = spec*random.random()
            rot2 = spec*random.random()
            rot1 = float(truncate(rot1, 5))
            rot2 = float(truncate(rot2, 5))
            cls.angles.append([rot1, rot2])

        model = int(str(cls.key)[:2])
        multi = int(str(cls.key)[2:4])

        mr1 = spec*random.random()
        mr2 = spec*random.random()
        mr1 = float(truncate(mr1, 5))
        mr2 = float(truncate(mr2, 5))

        posvec = []
        extra = 0
        for x in range(1,33):
            digits = cls.hash[2*(x-1):2*x]
            ran1 = random.random()
            while ran1 > 0.95 or ran1 < 0.05:
                ran1 = random.random()
            ran2 = random.random()
            while ran2 > 0.95 or ran2 < 0.05:
                ran2 = random.random()

            rot1 = (int(digits[0], 16) * spec/16 + ran1 * spec/16)
            rot2 = (int(digits[1], 16) * spec/16 + ran2 * spec/16)

            rot11 = float(truncate(((mr1+rot1) % spec), 5))
            rot22 = float(truncate(((mr2+rot2) % spec), 5))
            cell = ((multi)*x) - extra
            pos = cell % len(cls.origins)
            while pos in posvec:
                extra = extra + 1
                cell = ((multi)*x) + extra
                pos = cell % len(cls.origins)

            posvec.append(pos)
            cls.angles[pos][0] = rot11
            cls.angles[pos][1] = rot22

        model = model % len(cls.origins)

        if model in posvec:
            x = 33
            cell = ((multi)*x) + extra
            pos = cell % len(cls.origins)
            cls.angles[pos] = cls.angles[model]


        cls.angles[model] = [mr1,mr2]

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

        #normalizing normals
        cls.normals =  cls.normals*(1/np.sqrt(np.add(cls.normals*cls.normals,1)))

        wf = open('../stl/FOGcode.stl', 'wb+')
        wf.write(_b("\0"*80))

        wf.write(_b(np.uint32(cls.numFaces*cls.numSpheres)))

        for i in range(cls.numFaces*cls.numSpheres):
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

    @classmethod
    def mergeSTL(cls,final):
        # find the max dimensions, so we can know the bounding box, getting the height,
        # width, length (because these are the step size)...
        def find_mins_maxs(obj):
            minx = maxx = miny = maxy = minz = maxz = None
            for p in obj.points:
                # p contains (x, y, z)
                if minx is None:
                    minx = p[stl.Dimension.X]
                    maxx = p[stl.Dimension.X]
                    miny = p[stl.Dimension.Y]
                    maxy = p[stl.Dimension.Y]
                    minz = p[stl.Dimension.Z]
                    maxz = p[stl.Dimension.Z]
                else:
                    maxx = max(p[stl.Dimension.X], maxx)
                    minx = min(p[stl.Dimension.X], minx)
                    maxy = max(p[stl.Dimension.Y], maxy)
                    miny = min(p[stl.Dimension.Y], miny)
                    maxz = max(p[stl.Dimension.Z], maxz)
                    minz = min(p[stl.Dimension.Z], minz)
            return minx, maxx, miny, maxy, minz, maxz

        def translate(_solid, step, padding, multiplier, axis):
            if 'x' == axis:
                items = 0, 3, 6
            elif 'y' == axis:
                items = 1, 4, 7
            elif 'z' == axis:
                items = 2, 5, 8
            else:
                raise RuntimeError('Unknown axis %r, expected x, y or z' % axis)
            # _solid.points.shape == [:, ((x, y, z), (x, y, z), (x, y, z))]
            _solid.points[:, items] += (step * multiplier) + (padding * multiplier)

        # Using an existing stl file:
        main_body = mesh.Mesh.from_file('../stl/shuffledPart.stl', calculate_normals=False)
        # I wanted to add another related STL to the final STL
        code_body = mesh.Mesh.from_file('../stl/FOGcode.stl', calculate_normals=False)
        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(code_body)
        code_body.rotate([1,0,0],math.radians(cls.offset[0]))
        code_body.rotate([0,0,1],math.radians(cls.offset[1]))


        if final:
            translate(code_body, cls.newOrigin[0], cls.newOrigin[0] / 10., 1, 'x')
            translate(code_body, cls.newOrigin[1], cls.newOrigin[1] / 10., 1, 'y')
            translate(code_body, cls.newOrigin[2], cls.newOrigin[2] / 10., 1, 'z')
            print('Generated!')

        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(code_body)

        sign = 1
        if (maxx + minx)/2 < 0:
            sign = - 1
        cls.dispData[0] = sign* (maxx - minx)
        sign = 1
        if (maxy + miny)/2 < 0:
            sign = - 1
        cls.dispData[1] = sign * (maxy - miny)
        sign = 1
        if (maxz + minz)/2 < 0:
            sign = - 1
        cls.dispData[2] = sign * (maxz - minz)

        cls.combined = mesh.Mesh(np.concatenate([main_body.data, code_body.data]), calculate_normals=False)
        pp(cls.combined.data)
        cls.combined.save('../stl/combined.stl', mode=stl.Mode.BINARY, update_normals=False)  # save as ASCII

        # cls.combined = np.concatenate([main_body.data, code_body.data])


    @classmethod
    def codeplacementGUI(cls):

        def genView(ang1,ang2,x0,y0,z0):
            figure = pyplot.figure()
            axes = mplot3d.Axes3D(figure)

            fig = pyplot.figure()
            axes = fig.add_subplot(111, projection='3d')
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = abs(cls.dispData[0]/1.8) * np.outer(np.cos(u), np.sin(v))
            y = abs(cls.dispData[1]/1.8) * np.outer(np.sin(u), np.sin(v))
            z = abs(cls.dispData[2]/1.8) * np.outer(np.ones(np.size(u)), np.cos(v))

            #axes.plot_surface(x, y, z,  rstride=4, cstride=4, color='r')
            axes.plot_surface(x+cls.dispData[0]/2+x0, y+cls.dispData[1]/2+y0, z+cls.dispData[2]/2+z0,  rstride=4, cstride=4, color='r')

            # Load the STL files and add the vectors to the plot

            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(cls.combined.vectors))

            # Auto scale to the mesh size
            scale = cls.combined.points.flatten(-1)
            axes.auto_scale_xyz(scale, scale, scale)
            axes.view_init(elev=ang1, azim=ang2)
            axes.set_proj_type('ortho')
            axes.set_xlabel('X axis')
            axes.set_ylabel('Y axis')
            axes.set_zlabel('Z axis')

            # Show the plot to the screen
            pyplot.show(figure)

        cls.newOrigin=np.zeros(3)
        x=0
        y=0
        z=0

        clicks = 1
        while not clicks == 0:
             # genView(90,90,x,y,z)
             clicks = float(input('Move how far in the x directon...'))
             x = x + clicks
        cls.newOrigin[0]=x

        clicks = 1
        while not clicks == 0:
            # genView(0,0,x,y,z)
            clicks = float(input('Move how far in the y directon...'))
            y = y + clicks
        cls.newOrigin[1]=y

        clicks = 1
        while not clicks == 0:
            # genView(0,90,x,y,z)
            clicks = float(input('Move how far in the z directon...'))
            z = z + clicks
        cls.newOrigin[2]=z


def main():
    parser = argparse.ArgumentParser(description='DMX code embedder')
    parser.add_argument("filename", help="The filename of the STL")
    args = parser.parse_args()

    if (not os.path.exists(args.filename)):
	    print ("File", args.filename, "does not exist.")
	    sys.exit(1)

    # create mesh object
    mesh = generator()
    # generate key
    mesh.genKey()
    # shuffle the inputted part
    mesh.shufflePart(filename=args.filename)
    # generate hash from shuffled part
    mesh.genHash()
    # generate DataMatrix
    mesh.genMatrix()
    # reading matrix
    mesh.readMatrix()
    # x,y coordinates for matrix cells
    generator.CODExy()
    # x,y coordinates for FOG
    mesh.FOGxy()
    # z coordinates for each x,y
    mesh.assignZ()
    # encode hash in rotation
    mesh.genRotVal()
    # read model sphere from matlab data
    mesh.readModel()

    # store number of spheres
    generator.numSpheres = len(mesh.origins)

    # convert origins and angles from python list to np array
    generator.origins = np.array(generator.origins)
    generator.angles = np.array(generator.angles)


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

    final=False
    mesh.mergeSTL(final)
    mesh.codeplacementGUI()
    final=True
    mesh.mergeSTL(final)
    # # FOR DEBUGGING DECODER PURPOSES
    # np.savetxt("../legacy/angles.txt", mesh.angles)

if __name__ == '__main__':
    main()
