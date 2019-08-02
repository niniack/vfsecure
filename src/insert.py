# !/usr/bin/env python3
# insert.py v1 - script to insert a FOG code into a part
# by Nishant Aswani @niniack
# Composite Materials and Mechanics Laboratory, Copyright 2019

from pprintpp import pprint as pp
from utils import _b
import numpy as np
import math
import stl
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

HEADER_COUNT = 80

class insert():
    code=None
    part=None
    offset = [45,45]

    @classmethod
    def insertCode(cls):
        cf = open('../stl/sphere.stl', 'rb')

        dtObj = np.dtype([
        		('normals', np.float32, (3,)),
        		('vertices', np.float32, (3, 3)),
        		('attrs', np.uint16, (1,))
				])

        header = np.fromfile(cf, dtype=np.uint8, count=HEADER_COUNT)
        codeTriangles = np.fromfile(cf, dtype=np.uint32, count=1)
        code = np.fromfile(cf, dtype=dtObj, count=-1)
        cf.close()

        pf = open('../stl/knob.stl', 'rb')

        header =  np.fromfile(pf, dtype=np.uint8, count=HEADER_COUNT)
        partTriangles = np.fromfile(pf, dtype=np.uint32, count=1)
        part = np.fromfile(pf, dtype=dtObj, count=-1)
        pf.close()

        numTri = codeTriangles + partTriangles

        codeNormals = code['normals']
        codeVertices = code['vertices']

        partNormals = part['normals']
        partVertices = part['vertices']


        wf = open('merge.stl', 'wb+')
        wf.write(_b("\0"*80))

        wf.write(_b(np.uint32(numTri)))


        for i in range(int(partTriangles)):
            wf.write(_b(np.float32(partNormals[i][0])))
            wf.write(_b(np.float32(partNormals[i][1])))
            wf.write(_b(np.float32(partNormals[i][2])))

            for j in range(3):
            	wf.write(_b(np.float32(partVertices[i][j][0])))
            	wf.write(_b(np.float32(partVertices[i][j][1])))
            	wf.write(_b(np.float32(partVertices[i][j][2])))

            wf.write(_b(np.uint16(0)))

        for i in range(int(codeTriangles)):
            wf.write(_b(np.float32(codeNormals[i][0])))
            wf.write(_b(np.float32(codeNormals[i][1])))
            wf.write(_b(np.float32(codeNormals[i][2])))

            for j in range(3):
            	wf.write(_b(np.float32(codeVertices[i][j][0])))
            	wf.write(_b(np.float32(codeVertices[i][j][1])))
            	wf.write(_b(np.float32(codeVertices[i][j][2])))

            wf.write(_b(np.uint16(0)))



        wf.close()

    @classmethod
    def mergeSTL(cls,x,y,z):
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
        main_body = mesh.Mesh.from_file('../stl/knob.stl')
        print(main_body)

        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(main_body)
        w1 = x*(maxx - minx)
        l1 = y*(maxy - miny)
        h1 = z*(maxz - minz)


        # I wanted to add another related STL to the final STL
        code_body = mesh.Mesh.from_file('../stl/FOGcode.stl')
        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(code_body)
        code_body.rotate([1,0,0],math.radians(cls.offset[0]))
        code_body.rotate([0,0,1],math.radians(cls.offset[1]))

        w2 = maxx - minx
        l2 = maxy - miny
        h2 = maxz - minz

        translate(code_body, w1, w1 / 10., -1, 'x')
        translate(code_body, l1, l1 / 10., -1, 'y')
        translate(code_body, h1, h1 / 10., 1, 'z')
        combined = mesh.Mesh(np.concatenate([main_body.data, code_body.data]))


        combined.save('../stl/combined.stl', mode=stl.Mode.ASCII)  # save as ASCII

        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)

        # Load the STL files and add the vectors to the plot

        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(combined.vectors))

        # Auto scale to the mesh size
        scale = combined.points.flatten(-1)
        axes.auto_scale_xyz(scale, scale, scale)
        axes.view_init(elev=90, azim=90)
        axes.set_proj_type('ortho')
        axes.set_xlabel('X axis')
        axes.set_ylabel('Y axis')
        axes.set_zlabel('Z axis')

        # Show the plot to the screen
        pyplot.show()


def main():
    obj = insert()
    user = False
    x = 0
    y = 0
    z = 0
    while not user:
        obj.mergeSTL(x,y,z)
        user = int(input('Is there correct placement...  '))
        if not user:
            dir = int(input('What direction x=1, y=2, z=3...  '))
            if dir == 1:
                x=float(input("How far in the x direction"))
            if dir == 2:
                y=float(input("How far in the y direction"))
            if dir == 3:
                z=float(input("How far in the z direction"))



if __name__ == '__main__':
    main()
