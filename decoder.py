# !/usr/bin/env python3
# decoder.py v1 - Python script to extract separate entities from each other in an STL file
# by Nishant Aswani @niniack
# Composite Materials and Mechanics Laboratory, Copyright 2019

import argparse
import os
import struct
import sys
import numpy as np
from stl import mesh
from pprintpp import pprint as pp

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

HEADER_COUNT = 80
TRI_BYTE_COUNT = 4
DATA_COUNT = 3
ATTR_COUNT = 1
# import cv2
# from pyzbar.pyzbar import decode

class load:

	numTriangles=None
	filename=None
	normals=None
	vectors=None
	attrs=None
	size=None
	mem = np.array([[[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]])
	numForms=0

	@classmethod
	def readSTL(cls, filename):

		print ("Reading STL File")
		df = open(filename,'rb')
		header = df.read(80)
		df.close()

		if (header[0:5] == "solid"):
			print
			readText(filename)

		else:
			cls.readBinary(filename)

	@classmethod
	def readBinary(cls,filename):

		dtObj = np.dtype([
        		('normals', np.float32, (3,)),
        		('vectors', np.float32, (3, 3)),
        		('attrs', np.uint16, (1,))
				])

		df = open(filename,'rb')
		# cls.size = df.seek(0,os.SEEK_END)
		# print("The file size is: " + str(cls.size))
		# df.seek(0)

		header = np.fromfile(df, dtype=np.uint8, count=HEADER_COUNT)
		cls.numTriangles = np.fromfile(df, dtype=np.uint32, count=1)
		mesh = np.fromfile(df, dtype=dtObj, count=-1)
		df.close()

		cls.normals = mesh['normals']
		cls.vectors = mesh['vectors']
		cls.attr = mesh['attrs']

	# @classmethod
	# def findOpposites(cls):
	# 	# absNormals = np.absolute(cls.normals)
	# 	# uniqueNormals = np.unique(cls.normals)
	# 	# uniqueAbsNormals = np.unique(absNormals, axis=0)
	# 	# print("Unique Absolute Normals:" + str(uniqueAbsNormals.size))
	# 	# print("Unique Normals:" + str(uniqueNormals.size))
	# 	# print("Normals:" + str(cls.normals.size))
	#
	# 	# a = np.array([0,0,0])
	# 	# b = np.array([[0,0,0], [1,2,1], [3,4,3], [4,3,1]])
	# 	#
	# 	# result = np.all(a == b, axis=(1))
	# 	# print(result)
	# 	negativeNormals = np.negative(cls.normals)
	# 	opposites  = []
	#
	# 	for i in range(int(cls.numTriangles)-14000):
	# 		result = np.asarray(np.where(np.all(negativeNormals[i] == cls.normals, axis=1)))
	# 		opposites.append([i,result])
	#
	# 	print(opposites)

	@classmethod
	def findForm(cls, rowloc):

		#select a starting vertex to branch from, store as a np.ndarray
		vertices = cls.vectors
		form = []

		model = vertices[rowloc][0]
		# pp(model)
		form, vertices = cls._verticesHandler(model, vertices, form)
		# pp(len(form))
		i=1
		while i in range(len(form)):
			for j in range(3):
				model = form[i][j]
				form, vertices = cls._verticesHandler(model, vertices, form)
			i += 1

		cls.mem = np.append(cls.mem,form, axis=0)
		cls.numForms += 1
		pp(cls.numForms)
		return form

	@classmethod
	def _verticesHandler(cls, model, vertices, form):

		# find the indices where the model exists
		rows,columns = np.where(np.all(model == vertices, axis=2))

		# for the number of elements found, iterate through the vertices at
		# at relevant indices, copy those values and then delete them
		for j in range(rows.size):
			temp = vertices[rows[j]].tolist()
			form.append(temp)

		vertices = np.delete(vertices, rows, 0)
		return form, vertices

	@classmethod
	def checkForm(cls, rowloc):
		# cls.mem[0] = cls.vectors[rowloc].tolist()
		formExist = np.any(np.all(cls.vectors[rowloc] == cls.mem, axis=2))
		return formExist



def plotForm(form, fig, ax):

	# pp(len(form))
	xdata = []
	ydata = []
	zdata = []

	for i in range(len(form)):
		for j in range(3):
			xdata.append(form[i][j][0])
			ydata.append(form[i][j][1])
			zdata.append(form[i][j][2])

	ax.scatter(xdata, ydata, zdata, c=zdata)

	# pp("the number of data points are: "+ str(len(xdata)))



# def findCircumcenter(vectors,dim):
#
# 	triOrigins = np.empty([dim,3])
#
# 	for i in range(dim):
# 		triOrigins[i][0] = (vectors[i][0][0]+vectors[i][1][0]+vectors[i][2][0])/3
# 		triOrigins[i][1] = (vectors[i][0][1]+vectors[i][1][1]+vectors[i][2][1])/3
# 		triOrigins[i][2] = (vectors[i][0][2]+vectors[i][1][2]+vectors[i][2][2])/3
#
# 	return triOrigins
#
# def plotNormals(origins,normals,dim):
#
# 	x = np.empty([dim,1])
# 	y = np.empty([dim,1])
# 	z = np.empty([dim,1])
# 	u = np.empty([dim,1])
# 	v = np.empty([dim,1])
# 	w = np.empty([dim,1])
#
# 	fig = plt.figure()
# 	ax = fig.gca(projection='3d')
#
#
#
# 	for i in range(dim):
# 		x[i] = origins[i][0]
# 		y[i] = origins[i][1]
# 		z[i] = origins[i][2]
# 		u[i] = normals[i][0]
# 		v[i] = normals[i][1]
# 		w[i] = normals[i][2]
#
# 	ax.quiver(x,y,z,u,v,w,length=0.01, normalize=True)
#
# 	plt.show()


def main():

	parser = argparse.ArgumentParser(description="QR Code Extractor")
	parser.add_argument("filename", help="The filename of the STL")
	args = parser.parse_args()

	if (not os.path.exists(args.filename)):
	    print ("File", args.filename, "does not exist.")
	    sys.exit(1)

	mesh = load()
	mesh.readSTL(args.filename)

	# fig = plt.figure()
	# ax = fig.gca(projection='3d')

	for rowloc in range(int(mesh.numTriangles)):
		formExist = mesh.checkForm(rowloc)
		if (formExist == False):
			form = mesh.findForm(rowloc)
			# if (rowloc > 1):
			# 	plotForm(form, fig, ax)

	# plt.show()
	pp(mesh.numForms)

	# origins = findCircumcenter(mesh.vectors, int(mesh.numTriangles))
	# plotNormals(origins, normals, int(mesh.numTriangles))


if __name__ == '__main__':
    main()
