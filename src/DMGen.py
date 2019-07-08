#!/usr/bir/env python3
#python3 ./dmgen.pv

from matplotlib.image import imread, imsave
import numpy as np
import operator
import math
import random
import matplotlib.pyplot as plt

img = imread('barcode.png')

img = img[:,:,0]
hgt = len(img)
if not img[0,0]:
    img = np.logical_not(img)
cellsize = 1
col = hgt-1 #minus one because 0 is 1
while not img[0,col-1]:
    cellsize = cellsize + 1
    col = col - 1

row = 0
col = 0
row1 = 0
col1 = 0
Resizerow = []
ResizeMTX = []
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


a = np.array(ResizeMTX)
codedim = a.shape[1]
factor = 2#input("Fog Factor:")
fogdim = factor*codedim
fograd = fogdim/2

offset = math.sqrt(2) * codedim/2
x = 0
y = 0

while ((x-codedim)**2)+((y-codedim)**2) > math.sqrt(codedim**2-offset):
    x = random.randint(0,fogdim)
    y = random.randint(0,fogdim)

x1=x
y1=y
# if x <= codedim - cellsize/2:
#     x = cellsize * math.ceil(x/cellsize) + cellsize/2
#     print('c') #less than 8
#
# if y <= codedim - cellsize/2:
#     y = cellsize * math.ceil(y/cellsize) + cellsize/2
#     print('d') #less than 8
#
# if x > codedim - cellsize/2:
#     x = cellsize * math.floor(x/cellsize) + cellsize/2
#     print('a')
#
# if y > codedim - cellsize/2:
#     y = cellsize * math.floor(y/cellsize) + cellsize/2
#     print('b')


leftbuff = x - codedim/2
rightbuff = fogdim - leftbuff
axis = 0

pad_size = rightbuff - a.shape[axis]
axis_nb = len(a.shape)
if pad_size >= 0:
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (int(leftbuff), int(pad_size))
    a = np.pad(a, pad_width=npad, mode='constant', constant_values=0)


leftbuff = y - codedim/2
rightbuff = fogdim - leftbuff
axis = 1

pad_size = rightbuff - a.shape[axis]
axis_nb = len(a.shape)
if pad_size >= 0:
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (int(leftbuff), int(pad_size))
    b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)


#print(b,b.shape)
density = 0.5
placed = []
count = 1
#for y in range(1,fogdim):
noc = 158
coords = []
redo = 1
for i in range(noc):
    x = random.randint(0,fogdim)
    y = random.randint(0,fogdim)

    XY = [x,y]
    redo = 0
    if XY in coords:
        count = count + 1
        print(count)
        redo = 1
    while ((x-10)**2) + ((y-10)**2) > 100 or redo:
        x = random.randint(0,fogdim)
        y = random.randint(0,fogdim)
        XY = [x, y]
        redo = 0
        if XY in coords:
            count = count + 1
            print(count)
            redo = 1

    coords.append(XY)
xvals=[]
for i in range(len(coords)):
    xvals.append(coords[i][0])

yvals=[]
for j in range(len(coords)):
    yvals.append(coords[j][1])

print(xvals)
print(yvals)




plt.scatter(xvals, yvals, marker='o')
plt.show()
# show color scale
# for i in range(noc):
#     while ((x-10)**2) + ((y-10)**2) > 100  and  count < 100 and x not in placed:
#         x = random.randint(0,fogdim)
#         count = count + 1
#     placed.append(x)
#     if count > 99:
#         print('hello')
#     print(x,y)
