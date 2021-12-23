"""
Read ``.obj`` files and store vertices and triangles. 
Quad mesh is not supported right now.
Normals, texture coordinates are not extracted right now. 
"""

import numpy as np
import sys

class Objfile:
    m_numVertices = 0
    m_numFaces = 0
    m_vertices = []
    m_indices = []
    m_filename = ""
    m_AABB = [0.0,0.0,0.0,0.0] # x_min, y_min, x_max, y_max
    w, h = 0.0, 0.0
    def read(self, filename=""):
        """
            Read ``.obj`` file created by Blender
        """
        self.m_filename = filename
        if filename == "":
            print("please give me the obj file name")
            sys.exit(1)

        with open(filename,"r") as file:
            lines = [line.strip("\n") for line in file.readlines()]
            for line in lines:
                if '#' in line or 'mtl' in line or 'o' in line:
                    continue
                x = line.split(" ")
                if x[0] == 'v': # vertex
                    pos = x[1:]
                    pos = [float(p) for p in pos]
                    self.m_vertices.append(pos)
                    self.m_numVertices += 1
                elif x[0] == 'vt':
                    continue
                elif x[0] == 'vn':
                    continue
                elif x[0] == 'f': #face
                    if len(x) == 5:
                        print("please use triangle mesh")
                        sys.exit(1)
                    tri =[int(xx) for xx in x[1:]]
                    self.m_indices.append(tri)
                    self.m_numFaces += 1
            file.close()
    def readTxt(self, filename=""):
        """
            Read 2D Mesh generate by Miles Macklin's program
        """
        self.m_filename = filename
        if filename == "":
            print("please give me the obj file name")
            sys.exit(1)

        with open(filename,"r") as file:
            point = False
            triangle = False
            lines = [line.strip("\n") for line in file.readlines()]
            for line in lines:
                if 'numPoints' in line:
                    point = True
                    triangle = False
                    continue
                elif 'numTriangle' in line:
                    triangle = True
                    point = False
                    continue

                if point:
                    pos = [float(p) for p in line.split(" ")]
                    self.m_vertices.append(pos)
                    self.m_numVertices += 1
                    continue
                elif triangle:
                    tri =[int(xx) for xx in line.split(" ")]
                    self.m_indices.append(tri)
                    self.m_numFaces += 1
                    continue
            file.close()

    def ouputObjfile(self):
        print(f"{self.m_numVertices } vertices, {self.m_numFaces} faces")
        for i in range(self.m_numVertices):
            print(i, " ", self.m_vertices[i])
        for i in range(self.m_numFaces):
            print(i, " ", self.m_indices[i])

    def getVertice(self):
        return np.asarray(self.m_vertices)
    def getFaces(self):
        return np.asarray(self.m_indices)
    def getNumVertice(self):
        return self.m_numVertices
    def getNumFaces(self):
        return self.m_numFaces

    def compute_AABB(self, positions):
        self.m_AABB[0] = np.min(positions[:,0])
        self.m_AABB[1] = np.min(positions[:,1])
        self.m_AABB[2] = np.max(positions[:,0])
        self.m_AABB[3] = np.max(positions[:,1])
        self.w = self.m_AABB[2] - self.m_AABB[0]
        self.h = self.m_AABB[3] - self.m_AABB[1]

    def normalized(self):
        self.positions = self.getVertice()
        self.compute_AABB(self.positions)
        center = lambda x, y: (x+y)/2.0
        center_x = center(self.m_AABB[2], self.m_AABB[0])
        center_y = center(self.m_AABB[3], self.m_AABB[1])
        self.positions -= np.array([center_x, center_y])
        max_scale = self.w if self.w > self.h else self.h 
        self.positions /= max_scale
        self.positions += np.array([1.0, 1.0])
        self.positions *= 0.5
        return self.positions

    def get_normalized_AABB(self):
        self.compute_AABB(self.positions)
        return np.asarray(self.m_AABB)

if __name__ == "__main__":
    objFile = Objfile()
    # objFile.read("2dMesh.obj")
    objFile.readTxt("bunny.txt")
    vertices = objFile.getVertice()
    print(vertices)
    faces = objFile.getFaces()
    print(faces)