#!/usr/bir/env python3
# https://barcode.tec-it.com/en/DataMatrix?data=Hi
 # generate a 10 by 10 code with no quiet space
#python3 ./dmgen.pv

from matplotlib.image import imread, imsave
import numpy as np
import operator
import math
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylibdmtx.pylibdmtx import encode

key = 1213
hash = "25210c83610ebca1a059c0bae8255eba2f95be4d1d7bcfa89d7248a82d9f111"
img = imread('../images/barcode.png')


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
fogdim = math.ceil(factor*codedim)
fograd = fogdim/2

offset = math.sqrt(2) * codedim/2
x = 0
y = 0

while ((x-(fogdim/2))**2)+((y-fogdim/2)**2) > math.sqrt((fogdim/2)**2-offset):
    x = random.randint(0,fogdim)
    y = random.randint(0,fogdim)

buff = 1

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
for p in range(fogdim):
    for o in range(fogdim):
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

count = 0

for t in range(len(coords)):
    x = coords[t][0]
    y = coords[t][1]
    z = random.randint(0,fogdim)

    while ((x-fogdim/2)**2) + ((y-fogdim/2)**2) + ((z-fogdim/2)**2) > (fogdim/2)**2 and count < 10000:
        count = count + 1
        z = random.randint(0,fogdim)
    count = 0
    coords[t].append(z)

def Sort(sub_li):
    sub_li.sort(key = lambda x: (x[0], x[1]))
    return sub_li

xvals = []
yvals = []
zvals = []
for i in range(len(coords)):
    xvals.append(coords[i][0])
    yvals.append(coords[i][1])
    zvals.append(coords[i][2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xvals, yvals, zvals, c='r', marker='o')
plt.show()

Sort(coords)
tf = open('Origins.txt', 'w+')
tf.write(str(coords))
tf.close()

#np.savetxt('origins.txt', coords, delimiter = ',')
##encode

model = int(str(key)[:2])
multi = int(str(key)[2:4])

mr1 = 360*random.random()
mr2 = 360*random.random()

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])
posvec=[]

extra = 0
for x in range(1,32+1):
    digits = hash[2*(x-1):2*x]
    rot1 = (mr1 + (int(digits[0], 16) * 180/16 - random.random() * 180/16)) % 360
    rot2 = (mr2 + (int(digits[1], 16) * 180/16 - random.random() * 180/16)) % 360
    rot1 = truncate(rot1, 5)
    rot2 = truncate(rot2, 5)
    cell = ((multi)*x)-1-extra
    if cell in posvec:
        extra = extra + 1
        cell = ((multi)*x)-1+extra
        print("WARNING")
    pos = cell % len(coords)
    posvec.append(pos)
    print(coords[pos])
    coords[pos] = [coords[pos] , rot1, rot2]
    print(coords[pos])
    print(len(posvec))





#plt.scatter(xvals, yvals, zvals, marker='s')
#plt.show()
