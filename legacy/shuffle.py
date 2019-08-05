# !/usr/bin/env python3
# shuffle.py v1 - Randomly shuffle the location of faces/vertices in a binary STL file
# by Nishant Aswani @niniack
# Composite Materials and Mechanics Laboratory, Copyright 2019


from pprintpp import pprint as pp
import random
import numpy as np

HEADER_COUNT = 80


def shuffleFV():

    dtObj = np.dtype([
        		('normals', np.float32, (3,)),
        		('vertices', np.float32, (3, 3)),
        		('attrs', np.uint16, (1,))
				])

    rf = open('../stl/knob.stl', 'r', encoding='ascii', errors='replace')
    header = np.fromfile(rf, dtype=np.uint8, count=HEADER_COUNT)
    numTri = int(np.fromfile(rf, dtype=np.uint32, count=1))
    part = np.fromfile(rf, dtype=dtObj, count=-1)
    rf.close()

    np.random.shuffle(part)




def main():
    shuffleFV();


if __name__ == '__main__':
    main()
