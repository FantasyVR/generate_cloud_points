# generate_cloud_points

Given a 2D triangle mesh, we could randomly generate cloud points that fill in the triangle mesh.

<p float="left">
<img src="data/bunny.png" width="200"> <img src="data/armadillo.png" width="200">  <img src="data/socks.png" width="200">
</p>

# Run
`python disp_mesh.py`

Or you could use command line arguments: `python disp_mesh.py -i data/socks.txt"

# Triangle Mesh
We provide three tirangle meshes: **bunny**, **armadillo** and **christmas socks**.

## Triangle mesh format
```
numPoints 100
0.1 0.2
....
numTriangles 200
0 1 2
....
```

# Implementation details
1. Read the triangle mesh with `read_obj.py`
2. Find boundary edges using `find_boundary.py`
3. Generate random points `p` in the AABB of triangle mesh
4. Shoot a ray from `p` and count how many time the ray intersects with boundary edges
5. If the number of intersections is even, then the point `p` is outside the triangle mesh
6. Otherwise, the point `p` is inside the triangle mesh
7. If `p` is inside the triangle mesh, we add it into the final cloud points field (`ti.Vector.field`)
