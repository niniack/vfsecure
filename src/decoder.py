# !/usr/bin/env python3
# decoder.py v1 - Python script to extract separate entities from each other in an STL file
# by Nishant Aswani @niniack
# Composite Materials and Mechanics Laboratory, Copyright 2019

import argparse
import os
import math
import struct
import sys
import numpy as np
from utils import _b
from pprintpp import pprint as pp

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

from itertools import chain
from collections import Counter

HEADER_COUNT = 80
TRI_BYTE_COUNT = 4
DATA_COUNT = 3
ATTR_COUNT = 1

class decoder():

	numTriangles = None
	filename = None
	normals = None
	vertices = None
	mem = None
	attrs = None
	size = None
	model = None
	numForms = 0
	multi = None
	unhash = None

	@classmethod
	def readSTL(cls, filename):

		print ("Reading STL File...")
		df = open(filename,'rb')
		header = df.read(80)
		df.close()

		# determine whether this is a Binary or ASCII file
		if (header[0:5] == "solid"):
			print("ASCII files not yet supported!")

		else:
			cls.readBinary(filename)

	@classmethod
	def readBinary(cls,filename):

		# Create data object to store values
		dtObj = np.dtype([
        		('normals', np.float32, (3,)),
        		('vertices', np.float32, (3, 3)),
        		('attrs', np.uint16, (1,))
				])

		df = open(filename,'rb')

		# Binary header
		header = np.fromfile(df, dtype=np.uint8, count=HEADER_COUNT)
		# Read number of triangles
		cls.numTriangles = np.fromfile(df, dtype=np.uint32, count=1)
		# Read remainder of data
		mesh = np.fromfile(df, dtype=dtObj, count=-1)

		df.close()

		# Save data to class variables
		cls.normals = mesh['normals']
		cls.vertices = mesh['vertices']
		cls.attr = mesh['attrs']

		# New array with vertex data and extra col to assign form number
		cls.mem = cls.vertices
		cls.mem = np.insert(cls.mem, 3, -1, axis=2)

	@classmethod
	def findForm(cls, rowloc, formTag):
		#
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
	def extractForm(cls, tag):
		rows, cols = np.where(cls.mem[:,:,3] == tag)
		rows = list(set(rows.tolist()))

		# BINARY FORMAT

		# UINT8[80] – Header
        # UINT32 – Number of triangles
        #
        #
        # foreach triangle
        # REAL32[3] – Normal vector
        # REAL32[3] – Vertex 1
        # REAL32[3] – Vertex 2
        # REAL32[3] – Vertex 3
        # UINT16 – Attribute byte count
        # end

		wf = open('cleanedPart.stl', 'wb+')
		wf.write(_b("\0"*80))

		wf.write(_b(np.uint32(len(rows))))

		for i in range(int(len(rows))):
			wf.write(_b(np.float32(cls.normals[rows[i]][0])))
			wf.write(_b(np.float32(cls.normals[rows[i]][1])))
			wf.write(_b(np.float32(cls.normals[rows[i]][2])))

			for j in range(3):
				wf.write(_b(np.float32(cls.vertices[rows[i]][j][0])))
				wf.write(_b(np.float32(cls.vertices[rows[i]][j][1])))
				wf.write(_b(np.float32(cls.vertices[rows[i]][j][2])))

			wf.write(_b(np.uint16(0)))

		wf.close()

	@classmethod
	def findPoleNormal(cls, modelTag):

		mRows, mColumns = np.where(cls.mem[:,:,3] == modelTag)
		mRows = list(set(mRows))
		mV = np.zeros((len(mRows), 3, 3))

		for i in range(len(mRows)):
			mV[i] = cls.vertices[mRows[i]]

		poles = cls._findPole(mV)

		# didn't bother use poles[0] because its the direct opposite
		# v1 = x,y,z
		# v2 = -x,-y,-z
		# so just invert it
		pRows, pCols = np.nonzero(np.all(mV == poles[1], axis=2))
		pRows = list(set(pRows.tolist()))



		poleFaceLocs = np.zeros(len(pRows))
		poleFaces = np.zeros((len(pRows), 3, 3))
		poleFaceNormals = np.zeros((len(pRows),3))

		for i in range(len(pRows)):
			poleFaceLocs[i] = mRows[pRows[i]]
			poleFaces[i] = cls.vertices[int(poleFaceLocs[i])]
			poleFaceNormals[i] = cls.normals[int(poleFaceLocs[i])]




		poleFaceNormals = poleFaceNormals[np.isfinite(poleFaceNormals)]
		poleFaceNormals = np.reshape(poleFaceNormals, [int(poleFaceNormals.size/3),3])
		poleVertexNormal = np.mean(poleFaceNormals, axis=0)

		return(poleVertexNormal)

		# origins = findCircumcenter(poleFaces, 14)
		# plotNormals(origins, poleFaceNormals, 14)
		# fig = plt.figure()
		# ax = fig.gca(projection='3d')
		# xdata = np.zeros(int(points.size/3))
		# ydata = np.zeros(int(points.size/3))
		# zdata = np.zeros(int(points.size/3))
		# for j in range(int(points.size/3)):
		# 	xdata[j] = points[j][0]
		# 	ydata[j] = points[j][1]
		#   zdata[j] = points[j][2]
		# ax.scatter(xdata, ydata, zdata, c=zdata)

	@classmethod
	def _findPole(cls, vertices):

		vertices = np.reshape(vertices, (int(vertices.size/3), 3))

		hits = 0
		poleLocs = np.zeros(2)
		poles = np.zeros([2,3])


		points, hits = np.unique(vertices, return_counts=True, axis=0)
		poleLocs[0] = hits.argmax()
		hits[int(poleLocs[0])] = -1;
		poleLocs[1] = hits.argmax()

		poles[0] = points[int(poleLocs[0])]
		poles[1] = points[int(poleLocs[1])]

		return poles
		# pp(hits)
		# return hits, highestHitRows

	@classmethod
	def findAngleDiff(cls, model, subject):

		# keep the other pole at hand, incase it is needed to switch
		nsubject = -subject

		# # rho value for spherical coordiantes
		# mRho = np.linalg.norm(model)
		# sRho = np.linalg.norm(subject)

		# Phi value of vectors (angle from the positive z-axis)
		# Use this value to determine which is the "correct" pole

		mPhi = np.rad2deg(np.arctan2(math.sqrt(math.pow(model[0],2)+math.pow(model[1],2)),model[2]))
		sPhi = np.rad2deg(np.arctan2(math.sqrt(math.pow(subject[0],2)+math.pow(subject[1],2)),subject[2]))
		nsPhi = np.rad2deg(np.arctan2(math.sqrt(math.pow(nsubject[0],2)+math.pow(nsubject[1],2)),nsubject[2]))

		# Theta value of vectors (angle from the positive x-axis)
		mTheta = np.rad2deg(np.arctan2(model[1], model[0])) + 90
		sTheta = np.rad2deg(np.arctan2(subject[1], subject[0])) + 90
		nsTheta = np.rad2deg(np.arctan2(nsubject[1], nsubject[0])) + 90

		# pp(mPhi)
		# pp(sPhi)
		# pp(nsPhi)

		# The smaller of the two phi differences is for the "correct" pole

		rotPhi = sPhi - mPhi
		rotTheta = sTheta - mTheta

		if rotPhi < 0:
		 	rotPhi = rotPhi + 90

		if rotTheta < 0:
		 	rotTheta = rotTheta + 90

		#pp(mPhi)
		#pp(sPhi)

		#pp("-----")

		#pp(mTheta)
		#pp(sTheta)

		# diffPhi2 = abs(mPhi - nsPhi)

		## LEGACY
		################################
		# if (diffPhi1 > diffPhi2):
		# 	rotPhi = diffPhi2
		# 	rotTheta = abs(mTheta-nsTheta)
		#
		# else:
		# 	rotPhi = diffPhi1
		# 	rotTheta = abs(mTheta-sTheta)
		################################

		return rotPhi, rotTheta

	@classmethod
	def findAngleDiffToAxes(cls, model):

		# # normal for x axis
		# x = np.array([0,0,1])
		# z = np.array([0,0,1])

		# conversion from cartesian to spherical formula
		mPhi = np.rad2deg(np.arctan2(math.sqrt(math.pow(model[0],2)+math.pow(model[1],2)),model[2]))
		mTheta = np.rad2deg(np.arctan2(model[1], model[0]))

		# xAxisPhi = np.rad2deg(np.arctan2(math.sqrt(math.pow(x[0],2)+math.pow(x[1],2)),x[2]))
		# zAxisTheta = np.rad2deg(np.arctan2(z[1], z[0]))
		rotPhi = mPhi
		rotTheta = mTheta+90

		# pp(xAxisPhi)
		# pp(zAxisTheta)
		return rotPhi, rotTheta

	@classmethod
	def readDMX(cls):
    	#model =  int(str(output)[:2])
		cls.mod = 4
    	#multi =  int(str(output)[2:4])
		cls.multi = 4

	@classmethod
	def readRot(cls):
        # read model offset from sphere
        # modelPhi, modelTheta = cls.findAngleDiffToAxes(model)
		def toHEX(f):
			if f<=10:
				h=str(f-1)
			if f>10:
				h=chr(97+(f-11))
			return h

		model = cls.findPoleNormal(cls.mod)

        # read all of the cells
		posvec = []
		cls.unhash = ''
		extra = 0

		for x in range(1,33):
			cell = ((cls.multi)*x) - extra
			pos = cell % cls.numForms
			while pos in posvec:
				extra = extra + 1
				cell = ((cls.multi)*x) + extra
				pos = cell % cls.numForms
			posvec.append(pos)

		if cls.mod in posvec:
			x = 33
			cell = ((cls.multi)*x) + extra
			pos = cell % cls.numForms
			posvec[(cls.mod in posvec)-1]=pos

		for y in range(len(posvec)):
			pos = posvec[y]
			subject = cls.findPoleNormal(pos)
			val = cls.findAngleDiff(model,subject)
			val1 = toHEX(math.ceil(val[0]*16/90))
			val2 = toHEX(math.ceil(val[1]*16/90))
			cls.unhash = cls.unhash + val1
			cls.unhash = cls.unhash + val2

	@classmethod
    def genHash(cls):
        cls.hash = hashlib.sha256(b'../stl/FOGcode.stl')

	@classmethod
    def compHash(cls):
		if cls.unhash == cls.hash:
			print('Authentic Orginal Part!')
		else
			print('Phoney Part')


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
	print("Finding forms...")
	for rowloc in range(int(mesh.numTriangles)): #mesh.numTriangles
		formExist = mesh.checkForm(rowloc)
		if (formExist == False):
			form = mesh.findForm(rowloc, formTag)
			formTag += 1
		# if(mesh.numForms > 20):
		# 	break
			# plotForm(form, fig, ax)
	# mesh.extractForm(20)



	########## DON'T DELETE ##################
	# model = mesh.findPoleNormal(cls.model)
	# subject = mesh.findPoleNormal(146)
	#
	# rotAngles = np.zeros()
	#
	# #rotPhi, rotTheta = mesh.findAngleDiff(model,subject)
	# modelPhi, modelTheta = mesh.findAngleDiffToAxes(model)

	mesh.readDMX()
	mesh.readRot()

	# pp("rotPhi: " + str(rotPhi))
	# pp("rotTheta: " + str(rotTheta))
	##########################################


	# plt.show()

	# origins = findCircumcenter(mesh.vectors, int(mesh.numTriangles))
	# plotNormals(origins, normals, int(mesh.numTriangles))


if __name__ == '__main__':
    main()



##### LEGACY ##### Useful to fish from sometimes

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

	# @classmethod
	# def triangleArea(cls, vertices):
	# 	# area = 1/2*(V x W)
	# 	v1 = vertices[1]-vertices[0]
	# 	v2 = vertices[2]-vertices[0]
	#
	# 	cross = np.cross(v1,v2)
	# 	mag = np.linalg.norm(cross)
	# 	area = (0.5)*mag
	# 	area = round(area,7)
	# 	return area
