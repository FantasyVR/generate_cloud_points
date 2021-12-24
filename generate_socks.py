import taichi as ti
import numpy as np
from read_obj import Objfile
import time
import argparse

ti.init(arch=ti.cpu, random_seed=int(time.time()))

parser = argparse.ArgumentParser(
    description='Generate Cloud Points in A object')
parser.add_argument('--input',
                    '-i',
                    nargs=1,
                    default="data/bunny.txt",
                    type=str,
                    required=False)
args = parser.parse_args()
input_file = args.input if type(args.input) is str else args.input[0]

objFile = Objfile()
objFile.readTxt(input_file)
NV = objFile.getNumVertice()
positions = objFile.normalized()
AABB = objFile.get_normalized_AABB()
bv = objFile.get_boundary_vertices()
be = objFile.get_boundary_edges()
NBE = objFile.get_num_boundary_edges()

NCP = 10000
cloud_points = ti.Vector.field(2, ti.f64, NCP)
num_cp = ti.field(ti.i32, ())
obj_pos = ti.Vector.field(2, ti.f64, NV)
boundary_edges = ti.Vector.field(2, ti.i32, NBE)


@ti.func
def isIntersectionX(p0, p1, x0):
    """Shoot a ray to x-axis from x0 and test if it intersect with (p0, p1) segment."""
    a, b = p1 - p0
    b0, b1 = x0 - p0
    t = b1 / b
    r = a * t - b0
    isInter = False
    if r >= 0 and 0 <= t <= 1:
        isInter = True
    return isInter


@ti.func
def generate_random_point(AABB: ti.template()):
    x, y = ti.random(), ti.random()
    w, h = AABB[2] - AABB[0], AABB[3] - AABB[1]
    w_p, h_p = 0.05 * w, 0.05 * h
    x_min, y_min = AABB[0], AABB[1]
    x = (w + w_p) * x + x_min - w_p * 0.5
    y = (h + h_p) * y + y_min - h_p * 0.5
    return ti.Vector([x, y])


@ti.func
def is_in_obj(random_point) -> ti.i32:
    is_in = True
    count = 0
    for i in range(NBE):
        idx0, idx1 = boundary_edges[i]
        p0, p1 = obj_pos[idx0], obj_pos[idx1]
        if isIntersectionX(p0, p1, random_point):
            count += 1
    if count % 2 == 0:
        is_in = False
    return is_in


@ti.func
def add_into_field(random_point):
    cloud_points[num_cp[None]] = random_point
    num_cp[None] += 1


@ti.kernel
def generate_socks():
    while num_cp[None] < NCP:
        random_point = generate_random_point(AABB)
        if is_in_obj(random_point):
            add_into_field(random_point)


obj_pos.from_numpy(positions)
boundary_edges.from_numpy(np.asarray(be, dtype=np.int32))
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
        bp = np.zeros(shape=(len(be), 2))
        ep = np.zeros(shape=(len(be), 2))
        for i in range(len(be)):
            bp[i] = positions[be[i][0]]
            ep[i] = positions[be[i][1]]
        gui.lines(bp, ep, radius=4, color=0x0000FF)
        boundary = np.zeros((len(bv), 2), dtype=np.float64)
        for i in range(len(bv)):
            boundary[i] = positions[bv[i]]
        gui.circles(boundary, radius=4, color=0x00FF00)

    generate_socks()
    gui.circles(cloud_points.to_numpy(), radius=1.0, color=0x00FF00)
    gui.rect([AABB[0], AABB[1]], [AABB[2], AABB[3]], radius=1, color=0xED553B)
    gui.show()
