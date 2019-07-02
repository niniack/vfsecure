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

class decoder:

	numTriangles=None
	filename=None
	normals=None
	vertices=None
	attrs=None
	size=None
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
        		('vertices', np.float32, (3, 3)),
        		('attrs', np.uint16, (1,))
				])

		df = open(filename,'rb')

		header = np.fromfile(df, dtype=np.uint8, count=HEADER_COUNT)
		cls.numTriangles = np.fromfile(df, dtype=np.uint32, count=1)
		mesh = np.fromfile(df, dtype=dtObj, count=-1)
		df.close()

		cls.normals = mesh['normals']
		cls.vertices = mesh['vertices']
		cls.attr = mesh['attrs']

		cls.mem = cls.vertices
		cls.mem = np.insert(cls.mem, 3, -1, axis=2)

	@classmethod
	def findForm(cls, rowloc, formTag):

		form = []
		model = cls.vertices[rowloc][0]

		cls._formDefiner(model, form,  formTag)

		i=1
		while i in range(len(form)):
			for j in range(3):
				model = form[i][j]
				form = cls._formDefiner(model, form, formTag)
			i += 1

		cls.numForms += 1
		pp(cls.numForms)
		return form

	@classmethod
	def _formDefiner(cls, model, form, tag):

		rows,columns = np.where(np.all(model == cls.vertices, axis=2))
		for j in range(rows.size):
			if (cls.mem[rows[j], 0, 3] == -1):
				temp = cls.vertices[rows[j]].tolist()
				form.append(temp)
				cls.mem[rows[j], :, 3] = tag

		return form

	@classmethod
	def checkForm(cls, rowloc):
		# cls.mem[0] = cls.vectors[rowloc].tolist()
		if (cls.mem[rowloc, 0, 3] != -1):
			formExist = True
		else:
			formExist = False

		return formExist

	@classmethod
	def findRotation(cls, modelTag, subjectTag):

		mRows, mVertices = np.where(cls.mem[:,:,3] == modelTag)
		sRows, sVertices = np.where(cls.mem[:,:,3] == subjectTag)

		# converted to set to remove duplicates, then back to list for indexing
		mRows = list(set(mRows))
		sRows = list(set(sRows))

		mV = cls.vertices[mRows[0]]


		mTri = cls.triangleArea(mV)
		pp(mTri)

		for row in range(len(sRows)):
			sV = cls.vertices[sRows[row]]
			sTri = cls.triangleArea(sV)
			pp(sTri)
			if (sTri == mTri):
				pp("wooden mouse")
				pp(row)



		# modelNormals = cls.normals[mRows[1]]
		# pp(modelNormals)


		# choose a reference sphere
		# select a sphere to compare with
		# choose a triangle on the reference sphere
		# find the same triangle on the other sphere using its area
		# separate function: calculate the diff in angle between the two vectors
		#
	@classmethod
	def triangleArea(cls, vertices):
		# area = 1/2*(V x W)
		v1 = vertices[1]-vertices[0]
		v2 = vertices[2]-vertices[0]

		cross = np.cross(v1,v2)
		mag = np.linalg.norm(cross)
		area = (0.5)*mag

		return area

def plotForm(form, fig, ax):

	xdata = []
	ydata = []
	zdata = []

	for i in range(len(form)):
		for j in range(3):
			xdata.append(form[i][j][0])
			ydata.append(form[i][j][1])
			zdata.append(form[i][j][2])

	ax.scatter(xdata, ydata, zdata, c=zdata)


def main():

	parser = argparse.ArgumentParser(description="QR Code Extractor")
	parser.add_argument("filename", help="The filename of the STL")
	args = parser.parse_args()

	if (not os.path.exists(args.filename)):
	    print ("File", args.filename, "does not exist.")
	    sys.exit(1)

	mesh = decoder()
	mesh.readSTL(args.filename)

	# fig = plt.figure()
	# ax = fig.gca(projection='3d')

	formTag = 0

	for rowloc in range(int(2300)):
		formExist = mesh.checkForm(rowloc)
		if (formExist == False):
			form = mesh.findForm(rowloc, formTag)
			formTag += 1
			# plotForm(form, fig, ax)
	mesh.findRotation(1,2)




	# origins = findCircumcenter(mesh.vectors, int(mesh.numTriangles))
	# plotNormals(origins, normals, int(mesh.numTriangles))


if __name__ == '__main__':
    main()



##### LEGACY #####

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
