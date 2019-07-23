from pprintpp import pprint as pp
import numpy as np
import math

from utils import _b


class generator:

    faces=None;
    vertices=None;
    origins=None;
    normals=None;
    shiftedVertices=None;
    numFaces=None;
    numVertices=None;



    @classmethod
    def readData(cls):
        # open two file buffers to read faces and vertices data
        vd = open('../sphere/vertices.txt','rb')
        fd = open('../sphere/faces.txt', 'rb')

        # load the buffers into arrays
        cls.vertices = np.loadtxt(vd,np.float32)
        cls.faces = np.loadtxt(fd,np.int32)

        # close the buffers to avoid data being overwritten or corrupted
        vd.close()
        fd.close()

        # convert from MATLAB indexing to Python indexing by subtracting 1
        cls.faces = np.subtract(cls.faces,1)

        cls.numFaces = np.size(cls.faces,0)
        cls.numVertices = np.size(cls.vertices,0)


    @classmethod
    def shift(cls, index):
        shifted = cls.vertices + cls.origins[index]
        numFaces = np.size(cls.faces,0)

        for i in range(numFaces):

            v0 = int(cls.faces[i][0])
            v1 = int(cls.faces[i][1])
            v2 = int(cls.faces[i][2])

            vec1 = shifted[v1] - shifted[v0]
            vec2 = shifted[v2] - shifted[v0]

            cls.normals[(numFaces*index)+i]=np.cross(vec1, vec2)

        for i in range(cls.numVertices):
            cls.shiftedVertices[(cls.numVertices*index)+i]=shifted[i]

    @classmethod
    def writeSTL(cls):

        numSpheres = np.size(cls.origins,0)

        #normalizing normals
        cls.normals =  cls.normals*(1/np.sqrt(np.add(cls.normals*cls.normals,1)))

        wf = open('sphere.stl', 'wb+')
        wf.write(_b("\0"*80))

        wf.write(_b(np.uint32(cls.numFaces*numSpheres)))

        for i in range(cls.numFaces*numSpheres):
            wf.write(_b(np.float32(cls.normals[i][0])))
            wf.write(_b(np.float32(cls.normals[i][1])))
            wf.write(_b(np.float32(cls.normals[i][2])))
            for j in range(3):

                sphereNum = math.floor(i/cls.numFaces)
                indicesFromSphereZero = cls.numFaces*sphereNum

                wf.write(_b(np.float32(cls.shiftedVertices[ cls.faces[i-indicesFromSphereZero][j]+(cls.numVertices*sphereNum) ][0])))
                wf.write(_b(np.float32(cls.shiftedVertices[ cls.faces[i-indicesFromSphereZero][j]+(cls.numVertices*sphereNum) ][1])))
                wf.write(_b(np.float32(cls.shiftedVertices[ cls.faces[i-indicesFromSphereZero][j]+(cls.numVertices*sphereNum) ][2])))

            wf.write(_b(np.uint16(0)))

        wf.close()


def main():
    mesh = generator()
    generator.readData()

    ######## these will be replaced #######
    generator.origins = np.loadtxt('../sphere/origins.txt',delimiter=',')
    numSpheres = np.size(generator.origins,0)
    #######################################

    generator.normals = np.zeros([generator.numFaces*numSpheres,3])
    generator.shiftedVertices = np.zeros([generator.numVertices*numSpheres,3])


    # for i in range(numSpheres):
    #     generator.shift(i)

    for i in range(numSpheres):
        generator.shift(i)

    pp(generator.shiftedVertices)
    generator.writeSTL()


if __name__ == '__main__':
    main()
