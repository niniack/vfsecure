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
    offset = [0,0]
    #
    # @classmethod
    # def insertCode(cls):
    #     cf = open('../stl/sphere.stl', 'rb')
    #
    #     dtObj = np.dtype([
    #     		('normals', np.float32, (3,)),
    #     		('vertices', np.float32, (3, 3)),
    #     		('attrs', np.uint16, (1,))
	# 			])
    #
    #     header = np.fromfile(cf, dtype=np.uint8, count=HEADER_COUNT)
    #     codeTriangles = np.fromfile(cf, dtype=np.uint32, count=1)
    #     code = np.fromfile(cf, dtype=dtObj, count=-1)
    #     cf.close()
    #
    #     pf = open('../stl/knob.stl', 'rb')
    #
    #     header =  np.fromfile(pf, dtype=np.uint8, count=HEADER_COUNT)
    #     partTriangles = np.fromfile(pf, dtype=np.uint32, count=1)
    #     part = np.fromfile(pf, dtype=dtObj, count=-1)
    #     pf.close()
    #
    #     numTri = codeTriangles + partTriangles
    #
    #     codeNormals = code['normals']
    #     codeVertices = code['vertices']
    #
    #     partNormals = part['normals']
    #     partVertices = part['vertices']
    #
    #
    #     wf = open('merge.stl', 'wb+')
    #     wf.write(_b("\0"*80))
    #
    #     wf.write(_b(np.uint32(numTri)))
    #
    #
    #     for i in range(int(partTriangles)):
    #         wf.write(_b(np.float32(partNormals[i][0])))
    #         wf.write(_b(np.float32(partNormals[i][1])))
    #         wf.write(_b(np.float32(partNormals[i][2])))
    #
    #         for j in range(3):
    #         	wf.write(_b(np.float32(partVertices[i][j][0])))
    #         	wf.write(_b(np.float32(partVertices[i][j][1])))
    #         	wf.write(_b(np.float32(partVertices[i][j][2])))
    #
    #         wf.write(_b(np.uint16(0)))
    #
    #     for i in range(int(codeTriangles)):
    #         wf.write(_b(np.float32(codeNormals[i][0])))
    #         wf.write(_b(np.float32(codeNormals[i][1])))
    #         wf.write(_b(np.float32(codeNormals[i][2])))
    #
    #         for j in range(3):
    #         	wf.write(_b(np.float32(codeVertices[i][j][0])))
    #         	wf.write(_b(np.float32(codeVertices[i][j][1])))
    #         	wf.write(_b(np.float32(codeVertices[i][j][2])))
    #
    #         wf.write(_b(np.uint16(0)))
    #
    #
    #
    #     wf.close()

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
        # I wanted to add another related STL to the final STL
        code_body = mesh.Mesh.from_file('../stl/FOGcode.stl')
        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(code_body)
        code_body.rotate([1,0,0],math.radians(cls.offset[0]))
        code_body.rotate([0,0,1],math.radians(cls.offset[1]))
        print("---------------------------------------")


        w2 = maxx - minx
        l2 = maxy - miny
        h2 = maxz - minz
        #translate(code_body, w2, w2 / 10., -1, 'x')
        #translate(code_body, l2, l2 / 10., -1, 'y')
        #translate(code_body, h2, h2 / 10., -1, 'z')
        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(code_body)
        w1 = maxx - minx
        l1 = maxy - miny
        h1 = maxz - minz

        combined = mesh.Mesh(np.concatenate([main_body.data, code_body.data]))


        combined.save('../stl/combined.stl', mode=stl.Mode.BINARY)  # save as ASCII

        def genView(ang1,ang2,x0,y0,z0):
            figure = pyplot.figure()
            axes = mplot3d.Axes3D(figure)


            fig = pyplot.figure()
            axes = fig.add_subplot(111, projection='3d')
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = (maxx-minx)/1.8 * np.outer(np.cos(u), np.sin(v))
            y = (maxy-miny)/1.8 * np.outer(np.sin(u), np.sin(v))
            z = (maxz-minz)/1.8 * np.outer(np.ones(np.size(u)), np.cos(v))
            axes.plot_surface(x+w1/2+x0, y+l1/2+y0, z+h1/2+z0,  rstride=4, cstride=4, color='r')
            print()
            # Load the STL files and add the vectors to the plot

            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(combined.vectors))



            # Auto scale to the mesh size
            scale = combined.points.flatten(-1)
            axes.auto_scale_xyz(scale, scale, scale)
            axes.view_init(elev=ang1, azim=ang2)
            axes.set_proj_type('ortho')
            axes.set_xlabel('X axis')
            axes.set_ylabel('Y axis')
            axes.set_zlabel('Z axis')

            # Show the plot to the screen
            pyplot.show(figure)


        newOrigin=np.zeros(3)
        x=0
        y=0
        z=0
        genView(90,90,x,y,z)
        clicks = 1
        while not clicks == 0:
             genView(90,90,x,y,z)
             clicks = float(input('Move how far in the x directon...'))
             x = x + clicks
        newOrigin[0]=x
        #
        # x=-0.001
        # y=0.034
        #
        # genView(0,0,x,y)
        # clicks = 1
        # while not clicks == 0:
        #     genView(0,0,x,y)
        #     clicks = float(input('Move how far in the y directon...'))
        #     x = x + clicks * 0.005
        #     yclicks = yclicks+clicks
        #
        #
        # genView(0,90,x,y)
        # clicks = 1
        # while not clicks == 0:
        #     genView(0,90,x,y)
        #     clicks = float(input('Move how far in the z directon...'))
        #     x = x + clicks * 0.005
        #     zclicks = zclicks+clicks

        #user = int(input('Move how far in the x directon...'))
        #while user:

            #genView(90,90,x,y)
            #user = int(input('Move how far in the x directon...'))



        #genView(90,0,x,y)

        #genView(0,90,x,y)


def main():
    obj = insert()
    user = False
    x = 1
    y = 1
    z = 1
    while not user:
        obj.mergeSTL(x,y,z)

        user = True#int(input('Is there correct placement...  '))
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
