# Utility script for cross-file tools

import numpy as np

def _b(s, encoding='ascii', errors='replace'):
	if isinstance(s, str):
		return bytes(s, encoding, errors)
	else:
		return s
def _standardWriteSTL(filename, numTri, normals, vertices):

	wf = open(filename, 'wb+')
	wf.write(_b("\0"*80))

	wf.write(_b(np.uint32(numTri)))

	for i in range(numTri):
		wf.write(_b(np.float32(normals[i][0])))
		wf.write(_b(np.float32(normals[i][1])))
		wf.write(_b(np.float32(normals[i][2])))

		for j in range(3):
			wf.write(_b(np.float32(vertices[i][j][0])))
			wf.write(_b(np.float32(vertices[i][j][1])))
			wf.write(_b(np.float32(vertices[i][j][2])))

		wf.write(_b(np.uint16(0)))

	wf.close()
