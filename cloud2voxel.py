import numpy as np
import pandas as pd
from numba import jit
import open3d as o3d
import cv2 as cv
from pyntcloud import PyntCloud

# Use jit compiler to speed up things
@jit(nopython=True)
def generate_voxels_numba(cloud, voxelgrid, voxels, num_points,\
                          grid_size, minxyz, max_voxels, max_num_points,\
                          voxel_size):

    # Initialize num_voxels
    num_voxels = 0
    for i in range(cloud.shape[0]):
        # Transform coords to voxel space
        cv = np.floor((cloud[i,:3] - minxyz.astype(np.float64)) / voxel_size.astype(np.float64)).astype(np.int32)

        # Ignore points outside of range
        if np.any(cv < np.array([0,0,0])) or np.any(cv >= grid_size):
            continue

        # Get corresponding voxel for each point
        voxelid = voxelgrid[cv[0], cv[1], cv[2]]

        # Case 1: new voxel
        if voxelid == -1:
            voxelid = num_voxels
            voxelgrid[cv[0], cv[1], cv[2]] = num_voxels
            num_voxels += 1

        # Case 2: existing voxel, check current number of points
        if num_points[voxelid] < max_num_points:
            voxels[voxelid, num_points[voxelid]] = cloud[i]
            num_points[voxelid] += 1

    # If the number of non-empty voxels exceeds max allowed, use random sampling.
    if num_points.nonzero()[0].shape[0] > max_voxels:
        # Prioratize voxels with more points
        idx = num_points.nonzero()[0] # Non zero voxels
        idx = np.argsort(num_points[idx])[::-1] # Sort ids: prioritize voxels with more points
        voxels = voxels[idx[0:max_voxels]]
    else:
        idx = num_points.nonzero()[0]
        inv_idx = np.nonzero(num_points == 0)[0]

        # Zero padding needed to maintain dimensions
        pd = max_voxels - idx.shape[0]
        idx = np.append(idx, inv_idx[0:pd])

        voxels = voxels[np.sort(idx)]

    # Final shape: Voxel x [xyz, sem2d, sem3d],
    #                      [xyz, sem2d, sem3d],
    #                      [xyz, sem2d, sem3d]

    return voxels

class Voxel2Cloud():
    def __init__(self, cloud, voxel_size, max_num_points, max_voxels, pc_range):

        self.cloud = cloud
        self.voxel_size = np.array(voxel_size)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.pc_range = pc_range

        # Voxels variables
        self.minxyz = np.array([self.pc_range[0], self.pc_range[2], self.pc_range[4]]).astype(int)
        self.grid_size_x = np.floor((self.pc_range[1] - self.pc_range[0]) / self.voxel_size[0]).astype(int)
        self.grid_size_y = np.floor((self.pc_range[3] - self.pc_range[2]) / self.voxel_size[1]).astype(int)
        self.grid_size_z = np.floor((self.pc_range[5] - self.pc_range[4]) / self.voxel_size[2]).astype(int)
        self.grid_size = np.array([self.grid_size_x, self.grid_size_y, self.grid_size_z]).astype(int)
        # Manually defined grid
        self.grid_size = np.array([600,200,30]).astype(int)

    def generate_voxels(self):

        # Compare the number of voxels with max allowed.
        # If it is higher, use it.
        if self.grid_size[0]*self.grid_size[1]*self.grid_size[2] > self.max_voxels:
            n_voxels = self.grid_size[0]*self.grid_size[1]*self.grid_size[2]
        # If it is lower use max number.
        else:
            n_voxels = self.max_voxels

        voxelgrid = -np.ones(self.grid_size, dtype=np.int32)
        voxels = np.zeros((n_voxels, self.max_num_points, 3))
        num_points = np.zeros(n_voxels, dtype=np.int32)

        # Call voxel generation with numba to improve performance
        voxels = generate_voxels_numba(self.cloud, voxelgrid, voxels,\
                                            num_points, self.grid_size, self.minxyz,\
                                            self.max_voxels, self.max_num_points, self.voxel_size)

        return voxels


def main():

    # Configuration
    # Numbers may vary depending on the cloud
    voxel_size = [0.2, 0.2, 0.2] # 20 cm in xyz
    max_num_points = 100
    max_voxels = 70000

    # Read cloud from file
    pcd = o3d.io.read_point_cloud('map.pcd')

    # Converto cloud to array
    cloud = np.asarray(pcd.points)

    # Get cloud boundaries
    pc_range = [np.min(cloud[:,0]), np.max(cloud[:,0]), np.min(cloud[:,1]), np.max(cloud[:,1]), np.min(cloud[:,2]), np.max(cloud[:,2])]

    # Voxel-like subsampling
    vx = Voxel2Cloud(cloud, voxel_size, max_num_points, max_voxels, pc_range)
    voxels = vx.generate_voxels()

    # Only use voxel centroids
    centroids = []

    for i in range(voxels.shape[0]):
        aux = voxels[i,np.where(voxels[i,:] > 0.0)[0], :]
        l = len(aux)

        if l > 0:
            centroids.append((sum(aux[:,0])/l,sum(aux[:,1])/l,sum(aux[:,2])/l))

    centroids = np.asarray(centroids)

    # Save centroids cloud to ply file
    centroids_cloud = PyntCloud(pd.DataFrame(data=centroids, columns=['x', 'y', 'z']))
    centroids_cloud.to_file('centroids.ply')

    test = o3d.io.read_point_cloud('centroids.ply')

    # Voxel mesh
    vox_mesh = o3d.geometry.TriangleMesh()
    for c in test.points:
       cube=o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2,
       depth=0.2)
       cube.paint_uniform_color((0.3,0.6,0.23))
       cube.translate(c, relative=False)
       vox_mesh+=cube

    # Compute normals to improve shading in visualization
    vox_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([vox_mesh])

    # Save final mesh
    o3d.io.write_triangle_mesh('voxel_mesh.ply', vox_mesh)


if __name__ == '__main__':
    main()
