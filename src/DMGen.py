#!/usr/bir/env python3
#python3 ./dmgen.pv

from matplotlib.image import imread, imsave
import numpy as np
import operator
import math
import random
from mpl_toolkits.mplot3d import Axes3D
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
factor = 1.5#input("Fog Factor:")
fogdim = factor*codedim
fograd = fogdim/2

offset = math.sqrt(2) * codedim/2
x = 0
y = 0

while ((x-(fogdim/2))**2)+((y-fogdim/2)**2) > math.sqrt((fogdim/2)**2-offset):
    x = random.randint(0,fogdim)
    y = random.randint(0,fogdim)

print(x,y)

buff = 0
dt = []
for xi in range(int(x-codedim/2)-buff,int(x+codedim-codedim/2)+buff):
    for yi in range(int(y-codedim/2)-buff,int(y+codedim-codedim/2)+buff):

        XYi = [xi,yi]
        dt.append(XYi)

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

codecoords = []
for p in math.ceil(range(fogdim)):
    for o in math.ceil(range(fogdim)):
        if b[p][o]:
            XYc = [p,o]
            codecoords.append(XYc)

count = 0
density = 0.5
noc =  math.floor(density *(fograd**2)*3.14)
coords = []
redo = 1

for i in range(noc):
    x = random.randint(0,fogdim)
    y = random.randint(0,fogdim)
    XY = [x,y]
    redo = 0
    if XY in coords:
        count = count + 1
        redo = 1
    while ((x-fogdim/2)**2) + ((y-fogdim/2)**2) > (fogdim/2)**2 or redo:
        x = random.randint(0,fogdim)
        y = random.randint(0,fogdim)
        XY = [x, y]
        redo = 0
        if XY in coords:
            count = count + 1
            redo = 1

    coords.append(XY)

for c in range(len(dt)):
    if dt[c] in coords:
        coords.remove(dt[c])

coords = coords + codecoords




for t in range(len(coords)):
    x = coords[t][0]
    y = coords[t][1]
    z = random.randint(0,fogdim)
    while ((x-fogdim/2)**2) + ((y-fogdim/2)**2) + ((z-fogdim/2)**2) > (fogdim/2)**2:
        z = random.randint(0,fogdim)
    coords[t].append(z)


print(coords)



xvals = []
for i in range(len(coords)):
    xvals.append(coords[i][0])

yvals = []
for j in range(len(coords)):
    yvals.append(coords[j][1])

zvals = []
for k in range(len(coords)):
    zvals.append(coords[k][2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xvals, yvals, zvals, c='r', marker='o')

plt.show()

print(len(coords))
#plt.scatter(xvals, yvals, zvals, marker='s')
#plt.show()
