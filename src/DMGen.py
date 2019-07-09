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
factor = 3#input("Fog Factor:")
fogdim = factor*codedim
fograd = fogdim/2

offset = math.sqrt(2) * codedim/2
x = 0
y = 0

while ((x-(fogdim/2))**2)+((y-fogdim/2)**2) > math.sqrt((fogdim/2)**2-offset):
    x = random.randint(0,fogdim)
    y = random.randint(0,fogdim)
print(x,y)
x = 15
y = 15

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

print(b)
codecoords = []
for p in range(fogdim):
    for o in range(fogdim):
        if b[p][o]:
            XYc = [o,p]
            codecoords.append(XYc)
count = 0
noc = 200
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
    while ((x-fogdim/2)**2) + ((y-fogdim/2)**2) > (fogdim/2)**2 or redo:
        x = random.randint(0,fogdim)
        y = random.randint(0,fogdim)
        XY = [x, y]
        redo = 0
        if XY in coords:
            count = count + 1
            print(count)
            redo = 1

    coords.append(XY)




dt = []

for xi in range(int(x-codedim/2),int(x+codedim-codedim/2)):
    for yi in range(int(y-codedim/2),int(y+codedim-codedim/2)):

        XYi = [xi,yi]
        dt.append(XYi)

#print(b,b.shape)
density = 0.5
placed = []
count = 1
#for y in range(1,fogdim):


for c in range(len(dt)):
    if dt[c] in coords:
        coords.remove(dt[c])

coords = coords + codecoords

print(coords)

xvals = []
for i in range(len(coords)):
    xvals.append(coords[i][0])

yvals = []
for j in range(len(coords)):
    yvals.append(coords[j][1])

plt.scatter(xvals, yvals, marker='o')
plt.show()
