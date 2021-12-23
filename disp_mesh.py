import taichi as ti 
import numpy as np
from read_obj import Objfile
from find_boundary import findBoudaryEdge, findBoundaryPoints

ti.init(arch=ti.cpu)

objFile = Objfile()
objFile.readTxt("armadillo.txt")
vertices = objFile.getVertice()
faces = objFile.getFaces()
positions = objFile.normalized()

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
        bv = findBoundaryPoints(faces, positions.shape[0]) # boundary vertices
        be = findBoudaryEdge(faces)
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


    gui.show()