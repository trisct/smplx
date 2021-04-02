# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import argparse

import numpy as np
import torch
from icecream import ic

import smplx

if __name__ == '__main__':
    model_folder = 'models'
    model_type = 'smpl'
    gender='neutral'
    num_betas=5
    
    plot_joints=True
    
    sample_shape=True
    sample_pose=True

    plotting_module='pyrender'
    use_face_contour=False

    print(model_type)
    model = smplx.create(model_folder, model_type=model_type, gender=gender)
    print(model)

    betas = None
    poses = None
    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    if sample_pose:
        poses = torch.zeros([1, model.NUM_JOINTS, 3], dtype=torch.float32)
        #poses[:, 2, 0] = 1.57
        #poses[:, 12, 0] = 1.57
        poses[:, 12, 2] = -.3
        poses[:, 13, 2] = .3
        poses[:, 15, 2] = -.9
        poses[:, 16, 2] = .9
        poses = poses.reshape(1, -1)
    
    output = model(betas=betas, body_pose=poses, return_verts=True, return_full_pose=True)
    ic(output)
    ic(output.vertices.shape)
    ic(dir(output))
    
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    ic(output.betas.shape)
    ic(output.body_pose.shape)
    ic(output.full_pose.shape) 
    #ic(output.get)
    ic(output.global_orient.shape)
    #ic(output.items)
    ic(output.joints.shape)
    #ic(output.keys)
    ic(output.transl)
    #ic(output.values)
    ic(output.vertices.shape)

    ic(vertices.shape)
    ic(joints.shape)

    ic(vertices.max(axis=0))
    ic(vertices.min(axis=0))

    bbox_extent = vertices.max(axis=0) - vertices.min(axis=0)
    ic(bbox_extent)


    if plotting_module == 'pyrender':
        import pyrender
        import trimesh
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)
    elif plotting_module == 'matplotlib':
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

        if plot_joints:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
        plt.show()
    elif plotting_module == 'open3d':
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            vertices)
        mesh.triangles = o3d.utility.Vector3iVector(model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        geometry = [mesh]
        if plot_joints:
            joints_pcl = o3d.geometry.PointCloud()
            joints_pcl.points = o3d.utility.Vector3dVector(joints)
            joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
            geometry.append(joints_pcl)

        o3d.visualization.draw_geometries(geometry)
    else:
        raise ValueError('Unknown plotting_module: {}'.format(plotting_module))
