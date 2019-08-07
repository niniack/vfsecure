# !/usr/bin/env python3
# decoder.py v1 - Python script to separate entities from each other in an STL file
# by Nishant Aswani @niniack and Michael Linares @michaellinares
# Composite Materials and Mechanics Laboratory, Copyright 2019

# input argument parser
import argparse

import os
import math
import struct
import sys
import re

# data library
import numpy as np

# commonly used functions
from utils import _b
from utils import _extractWriteSTL

# pretty printing
from pprintpp import pprint as pp

# stl manipulation library for volume count
from stl import mesh

# graphing library for snapping datamatrix
from mpl_toolkits import mplot3d
from matplotlib import pyplot

# image reading libraries for reading datamatrix
from pylibdmtx.pylibdmtx import decode
import cv2
import hashlib



# from itertools import chain
# from collections import Counter

HEADER_COUNT = 80
TRI_BYTE_COUNT = 4
DATA_COUNT = 3
ATTR_COUNT = 1

BUF_SIZE = 65536

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

	Mesh=None

	volumeTag=0
	largestVolume=0


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
		cls.dtObj = np.dtype([
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
		cls.Mesh = np.fromfile(df, dtype=cls.dtObj, count=-1)

		df.close()

		# Save data to class variables
		cls.normals = cls.Mesh['normals']
		cls.vertices = cls.Mesh['vertices']
		cls.attr = cls.Mesh['attrs']

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

		cls._findVolume(formTag)
		cls.numForms += 1
		pp(cls.numForms)
		return form

	@classmethod
	def _findVolume(cls, tag):
		rows, cols = np.where(cls.mem[:,:,3] == tag)
		rows = list(set(rows.tolist()))

		formMesh = np.zeros(len(rows), dtype=mesh.Mesh.dtype)
		for i in range(len(rows)):
			formMesh[i] = cls.Mesh[rows[i]]

		formMesh = mesh.Mesh(formMesh)
		volume, cog, inertia = formMesh.get_mass_properties()

		if volume > cls.largestVolume:
			cls.largestVolume = volume
			cls.volumeTag = tag

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

	# Use tag to extract values WITH that tag
	@classmethod
	def extractForm(cls, tag):
		rows, cols = np.where(cls.mem[:,:,3] == tag)
		rows = list(set(rows.tolist()))

		_extractWriteSTL(filename='../stl/cleanedPart.stl', normals = cls.normals, vertices = cls.vertices, rows = rows)

	# Use tag to extract values WITHOUT that tag
	@classmethod
	def extractCode(cls, tag):
		rows, cols = np.where(cls.mem[:,:,3] != tag)
		rows = list(set(rows.tolist()))

		_extractWriteSTL(filename='../stl/extractedCode.stl', normals = cls.normals, vertices = cls.vertices, rows = rows)

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

		# The smaller of the two phi differences is for the "correct" pole

		rotPhi = sPhi - mPhi
		rotTheta = sTheta - mTheta

		if rotPhi < 0:
		 	rotPhi = rotPhi + 90

		if rotTheta < 0:
		 	rotTheta = rotTheta + 90


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
		offset = [96,69]

		# Create a new plot
		figure = pyplot.figure()
		axes = mplot3d.Axes3D(figure, proj_type = 'ortho')

		# Load the STL files and add the vectors to the plot
		extractedCode = mesh.Mesh.from_file('../stl/extractedCode.stl')

		extractedCode.rotate([0,0,1],math.radians(-offset[1]))
		extractedCode.rotate([1,0,0],math.radians(-offset[0]))


		axes.add_collection3d(mplot3d.art3d.Poly3DCollection(extractedCode.vectors, facecolors='black', edgecolors='black'))

		# Auto scale to the mesh size
		scale = extractedCode.points.flatten(-1)
		axes.auto_scale_xyz(scale, scale, scale)
		# axes.autoscale()

		axes.view_init(elev=90, azim=0)

		# Show the plot to the screen
		# pyplot.show(figure)
		figure.savefig('../images/scannedSTL.png')

		result = str(decode(cv2.imread('../images/scannedSTL.png'))[0][0])
		result = re.findall("\d{2}",result)
		cls.mod = int(result[0])
		cls.multi = int(result[1])

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


		cls.mod = cls.mod %cls.numForms

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
		hasher = hashlib.sha256()
		with open('../stl/cleanedPart.stl', 'rb') as f:
		    while True:
		        data = f.read(BUF_SIZE)
		        if not data:
		            break
		        hasher.update(data)

		# cls.hash = hashlib.sha256(b'../stl/cleanedPart.stl').hexdigest()

	@classmethod
	def compHash(cls):
		if cls.unhash == cls.hash:
			print('Authentic Orginal Part!')
		else:
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

	object= decoder()
	object.readSTL(args.filename)

	formTag = 0
	print("Finding forms...")
	for rowloc in range(int(object.numTriangles)): #object.numTriangles
		formExist = object.checkForm(rowloc)
		if (formExist == False):
			form = object.findForm(rowloc, formTag)
			formTag += 1
		# if(object.numForms > 1):
		# 	break

	object.extractCode(object.volumeTag)
	object.extractForm(object.volumeTag)
	object.readDMX()
	object.readRot()
	object.genHash()
	object.compHash()

	########## DON'T DELETE ##################
	# model = object.findPoleNormal(cls.model)
	# subject = object.findPoleNormal(146)
	#
	# rotAngles = np.zeros()
	#
	# #rotPhi, rotTheta = object.findAngleDiff(model,subject)
	# modelPhi, modelTheta = object.findAngleDiffToAxes(model)

	# object.readDMX()
	# object.readRot()

	# pp("rotPhi: " + str(rotPhi))
	# pp("rotTheta: " + str(rotTheta))
	##########################################


if __name__ == '__main__':
    main()
