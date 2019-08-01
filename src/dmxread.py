# !/usr/bin/env python3
# dmxread.py v1 - script to read a stl file to extract datamatrix
# by Nishant Aswani @niniack
# Composite Materials and Mechanics Laboratory, Copyright 2019

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

from pylibdmtx.pylibdmtx import decode
import cv2


# # Create a new plot
# figure = pyplot.figure()
# axes = mplot3d.Axes3D(figure, proj_type = 'ortho')
#
# # Load the STL files and add the vectors to the plot
# your_mesh = mesh.Mesh.from_file('../stl/sphere.stl')
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors, facecolors='black', edgecolors='black'))
#
# # Auto scale to the mesh size
# scale = your_mesh.points.flatten(-1)
# axes.auto_scale_xyz(scale, scale, scale)
#
# axes.view_init(elev=90, azim=0)
#
# # Show the plot to the screen
# # pyplot.show(figure)
# figure.savefig('plot.png')

result = decode(cv2.imread('../images/genDMTX.png'))
print(result)
