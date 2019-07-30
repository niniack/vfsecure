# !/usr/bin/env python3
# insert.py v1 - script to insert a FOG code into a part
# by Nishant Aswani @niniack
# Composite Materials and Mechanics Laboratory, Copyright 2019

from pprintpp import pprint as pp
from utils import _b
import numpy as np

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





def main():
    obj = insert()
    obj.insertCode()


if __name__ == '__main__':
    main()
