import taichi as ti 
import numpy as np
from read_obj import Objfile
from find_boundary import findBoudaryEdge, findBoundaryPoints
import time 

ti.init(arch=ti.cpu, random_seed = int(time.time()))

objFile = Objfile()
faces = objFile.getFaces()
positions = objFile.normalized()
objFile.readTxt("bunny.txt")
NV = objFile.getNumVertice()
NF = objFile.getNumFaces()  # number of faces
AABB = objFile.get_normalized_AABB()
bv = findBoundaryPoints(faces, positions.shape[0])
be = findBoudaryEdge(faces)


"""
1. generate points in AABB
2. test if the point in the object
3. if in object, add into ti.field
4. show generated filed
"""
NCP = 1000
cloud_points = ti.Vector.field(2, ti.f32, NCP)
random_point = ti.Vector.field(2, ti.f32, ())
num_cp = ti.field(ti.i32, ())
obj_pos = ti.Vector.field(2, float, NV)
obj_f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
boundary_vertices = ti.Vector.field(2, float, len(bv))
boundary_edges = ti.Vector.field(2, int, len(be))
@ti.func
def isIntersectionX(p0, p1, x0):
    """Shoot a ray to x-axis from x0 and test if it intersect with (p0, p1) segment."""
    a, b = p1 - p0
    b0, b1 = x0 - p0
    t = b1 / b
    r = a * t - b0
    isInter = False
    if r > 0 and 0 < t < 1:
        isInter = True
    return isInter

@ti.kernel
def generate_random_point(AABB: ti.ext_arr()):
    x = ti.random()
    y = ti.random()
    x_min, y_min, x_max, y_max = AABB[0], AABB[1], AABB[2], AABB[3]
    x = (x_max - x_min) * x + x_min
    y = (y_max - y_min) * y + y_min
    random_point[None] = ti.Vector([x, y])
    # print(random_point[None])

@ti.kernel 
def is_in_obj() -> ti.i32:
    is_in = True
    count = 0
    for i in boundary_edges:
        idx0, idx1 = boundary_edges[i] 
        p0, p1 = boundary_vertices[idx0], boundary_vertices[idx1]
        if isIntersectionX(p0, p1, random_point[None]):
            count += 1
    if count % 2 == 0:
        is_in = False
    return is_in

@ti.kernel 
def add_into_field():
    cloud_points[num_cp[None]] = random_point[None]
    num_cp[None] += 1

def fill_obj():
    while num_cp[None] < NCP:
        generate_random_point(AABB)    
        if is_in_obj():
            add_into_field()
        print(f"hello:{num_cp[None]}")

@ti.kernel
def init_boundary_vertice(bv: ti.ext_arr()):
    for i in range(bv.shape[0]):
        boundary_vertices[i] = obj_pos[bv[i]]

obj_pos.from_numpy(positions)
obj_f2v.from_numpy(faces)

init_boundary_vertice(np.asarray(bv, dtype=np.int32))
boundary_edges.from_numpy(np.asarray(be))

gui = ti.GUI("Diplay christmas mesh", res=(800, 800))
pause = True
while gui.running:
    for e in gui.get_events():
        if e.key == gui.SPACE and gui.is_pressed:
            pause = not pause
        elif e.key == gui.ESCAPE:
            gui.running = False
    gui.circles(positions, radius=2.0, color=0xFF0000)

    if not pause:
        bp = np.zeros(shape=(len(be),2))
        ep = np.zeros(shape=(len(be),2))
        for i in range(len(be)):
            bp[i] = positions[be[i][0]]
            ep[i] = positions[be[i][1]]
        gui.lines(bp,ep, radius=4,color=0x0000FF)
        boundary=np.zeros((len(bv),2),dtype=np.float64)
        for i in range(len(bv)):
            boundary[i] = positions[bv[i]]
        gui.circles(boundary,radius=4, color=0x00FF00)

    fill_obj()
    gui.circles(cloud_points.to_numpy(), radius=1.0, color=0x00FF00)

    gui.show()