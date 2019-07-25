from pprintpp import pprint as pp
import numpy as np
import math

from utils import _b


class generator:

    faces=None
    vertices=None
    origins=None
    angles=None
    normals=None
    shiftedVertices=None
    rotatedVertices=None
    numFaces=None
    numVertices=None

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
    def shift(cls, index, rotateBool):
        shifted = cls.vertices + cls.origins[index]
        numFaces = np.size(cls.faces,0)

        if (rotateBool == True):
            shifted = cls.rotate(cls.origins[index], shifted, index)


        # if (rotateBool == True):
        #
        #     ## Background Info:
        #     # x = rho * sin(theta) * cos(phi)
        #     # y = rho * sin(theta) * sin(phi)
        #     # z = rho * cos(theta)
        #
        #     # Initializing rho array
        #     rho = np.zeros([np.size(shifted,0)])
        #     # Calculating the differences (e.g. x2-x1, y2-y1, z2-z1) and squaring them
        #     difference = (shifted-cls.origins[index])**2
        #
        #     # Calculating the sqrt of the sum
        #     for row in range(np.size(difference,0)):
        #         rho[row] = np.sqrt(np.sum(difference[row]))
        #
        #     rotated = np.zeros([np.size(cls.vertices,0),3])
        #
        #     rotated[:,0] = rho * np.sin(cls.angles[index][0]) * np.cos(cls.angles[index][1])
        #     rotated[:,1] = rho * np.sin(cls.angles[index][0]) * np.sin(cls.angles[index][1])
        #     rotated[:,2] = rho * np.cos(cls.angles[index][0])
        #
        #     shifted = shifted + rotated

        # Calculate face normals

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
    def rotate(cls, origin, vertices, index):

        # append a column of 1s to the vertices for matrix transformations
        rawVertices = np.hstack((vertices, np.ones([np.size(vertices, 0),1])))

        # Transformation matrix 1 to move vertex by diff between sphere origin and 0,0,0
        t1 = np.identity(4)
        t1[3,:] = np.hstack((-origin,1))

        ######################## ROTATION #########################3
        rollAngle = np.deg2rad(cls.angles[index][0])
        roll = np.array([[1, 0, 0, 0], [0, np.cos(rollAngle), np.sin(rollAngle), 0], [0, -np.sin(rollAngle), np.cos(rollAngle), 0], [0, 0, 0, 1]] )

        yawAngle = np.deg2rad(cls.angles[index][1])
        yaw = np.array([[np.cos(yawAngle), np.sin(yawAngle), 0, 0], [-np.sin(yawAngle), np.cos(yawAngle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ######################## ROTATION #########################


        # Transformatio matrix 2 to move vertex back to original location
        t2 = np.identity(4)
        t2[3,:] = np.hstack((origin,1))

        rawVertices = np.matmul(rawVertices, t1)
        rawVertices = np.matmul(rawVertices, roll)
        rawVertices = np.matmul(rawVertices, yaw)
        rawVertices = np.matmul(rawVertices, t2)


        rawVertices = np.delete(rawVertices, 3, 1)
        # pp(rawVertices)
        return rawVertices

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
    mesh.readData()

    ######## these will be replaced #######
    mesh.origins = np.loadtxt('../sphere/origins.txt')
    mesh.angles = np.loadtxt('../sphere/angles.txt')
    numSpheres = np.size(mesh.origins,0)
    #######################################

    mesh.normals = np.zeros([mesh.numFaces*numSpheres,3])
    mesh.shiftedVertices = np.zeros([mesh.numVertices*numSpheres,3])
    mesh.rotatedVertices = np.zeros([mesh.numVertices*numSpheres, 3])


    for i in range(numSpheres): #numSpheres
        mesh.shift(i, True)

    mesh.writeSTL()


if __name__ == '__main__':
    main()
