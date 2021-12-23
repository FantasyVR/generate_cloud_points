import taichi as ti 
import numpy as np
from read_obj import Objfile

ti.init(arch=ti.cpu)

objFile = Objfile()
objFile.readTxt("armadillo.txt")
vertices = objFile.getVertice()
faces = objFile.getFaces()

positions = objFile.normalized()

gui = ti.GUI("Diplay christmas mesh", res=(800, 800))

while gui.running:
    gui.circles(positions, radius=2.0, color=0xFF0000)
    gui.show()