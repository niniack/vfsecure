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

HEADER_COUNT = 80

class insert():
    code=None
    part=None

    @classmethod
    def insertCode(cls):
        cf = open('sphere.stl', 'rb')

        dtObj = np.dtype([
        		('normals', np.float32, (3,)),
        		('vertices', np.float32, (3, 3)),
        		('attrs', np.uint16, (1,))
				])

        header = np.fromfile(cf, dtype=np.uint8, count=HEADER_COUNT)
        codeTriangles = np.fromfile(cf, dtype=np.uint32, count=1)
        code = np.fromfile(cf, dtype=dtObj, count=-1)
        cf.close()

        pf = open('knob.stl', 'rb')

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
    def numpy11(cls):
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


        def copy_obj(obj, dims, num_rows, num_cols, num_layers):
            w, l, h = dims
            copies = []
            for layer in range(num_layers):
                for row in range(num_rows):
                    for col in range(num_cols):
                        # skip the position where original being copied is
                        if row == 0 and col == 0 and layer == 0:
                            continue
                        _copy = mesh.Mesh(obj.data.copy())
                        # pad the space between objects by 10% of the dimension being
                        # translated
                        if col != 0:
                            translate(_copy, w, w / 10., col, 'x')
                        if row != 0:
                            translate(_copy, l, l / 10., row, 'y')
                        if layer != 0:
                            translate(_copy, h, h / 10., layer, 'z')
                        copies.append(_copy)
            return copies

        # Using an existing stl file:
        main_body = mesh.Mesh.from_file('../images/knob.stl')
        print(main_body)

        # rotate along Y
        main_body.rotate([0.0, 0.5, 0.0], math.radians(90))

        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(main_body)
        w1 = maxx - minx
        l1 = maxy - miny
        h1 = maxz - minz
        copies = copy_obj(main_body, (w1, l1, h1), 2, 2, 1)

        # I wanted to add another related STL to the final STL
        twist_lock = mesh.Mesh.from_file('ball_and_socket_simplified_-_twist_lock.stl')
        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(twist_lock)
        w2 = maxx - minx
        l2 = maxy - miny
        h2 = maxz - minz
        translate(twist_lock, w1, w1 / 10., 3, 'x')
        copies2 = copy_obj(twist_lock, (w2, l2, h2), 2, 2, 1)
        combined = mesh.Mesh(numpy.concatenate([main_body.data, twist_lock.data] +
                                            [copy.data for copy in copies] +
                                            [copy.data for copy in copies2]))

        combined.save('combined.stl', mode=stl.Mode.ASCII)  # save as ASCII



def main():
    obj = insert()
    obj.numpy11()


if __name__ == '__main__':
    main()