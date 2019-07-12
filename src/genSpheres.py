from pprintpp import pprint as pp
import numpy as np

from utils import _b


class generator:

    faces=None;
    vertices=None;
    origins=None;
    normals=None;
    shiftedSpheres=None;


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

    @classmethod
    def writeSTL(cls):

        numOrigins = np.size(cls.origins,0)
        numFaces = np.size(cls.faces,0)

        #normalizing normals(?)
        cls.normals =  cls.normals*(1/np.sqrt(np.add(cls.normals*cls.normals,1)))

        wf = open('sphere.stl', 'wb+')
        wf.write(_b("\0"*80))

        wf.write(_b(np.uint32(int(numFaces))))

        for i in range(int(numFaces)):
            wf.write(_b(np.float32(cls.normals[i][0])))
            wf.write(_b(np.float32(cls.normals[i][1])))
            wf.write(_b(np.float32(cls.normals[i][2])))
            for j in range(3):
                wf.write(_b(np.float32(cls.vertices[int(cls.faces[i][j])][0])))
                wf.write(_b(np.float32(cls.vertices[int(cls.faces[i][j])][1])))
                wf.write(_b(np.float32(cls.vertices[int(cls.faces[i][j])][2])))

            wf.write(_b(np.uint16(0)))

        wf.close()


def main():
    mesh = generator()
    generator.readData()


    generator.origins = np.loadtxt('../sphere/origins.txt',delimiter=',')
    numSpheres = np.size(generator.origins,0)
    numFaces = np.size(generator.faces,0)
    generator.normals = np.zeros([numFaces*numSpheres,3])


    # for i in range(numSpheres):
    #     generator.shift(i)

    generator.shift(0)

    generator.writeSTL()


if __name__ == '__main__':
    main()
