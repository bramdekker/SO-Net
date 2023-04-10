import sys
import os
import os.path
import numpy as np
import open3d as o3d
import torch

from util import som

def som_saver_catenary_arches(root, rows, cols, gpu_ids, output_root):
    som_builder = som.SOM(rows, cols, 3, gpu_ids)
    
    # Prepare the point clouds as a 3xN pytorch tensor (N arrays with 3 coordinates!)
    file_list = os.listdir(root)
    for j, f in enumerate(file_list):
        if os.path.isdir(os.path.join(root, f)):
            continue

        # Read las point cloud into array -> [[x0, ..., xn], [y0, ..., yn, [z0, ..., zn]]]
        # f_las = laspy.read(os.path.join(root, f))
        f_pcd = o3d.io.read_point_cloud(os.path.join(root, f))
        pc_np = np.asarray(f_pcd.points).T # 3xN tensor -> [[x0, ..., xn], [y0, ..., yn, [z0, ..., zn]]]
        # pc_np = np.vstack((f_las.x, f_las.y, f_las.z)) # 3xN tensor -> [[x0, ..., xn], [y0, ..., yn, [z0, ..., zn]]]
        # pc_np = np.transpose(pc_np) # Nx3 array -> [[x0, y0, z0], ..., [xn, yn, zn]]

        sn_np = np.asarray(f_pcd.normals).T

        # Downsample point cloud to take 524288 random points.
        pc_sampled = pc_np[:, np.random.choice(pc_np.shape[1], 1048576, replace=False)] # 3 x sample_size
        pc = torch.from_numpy(pc_sampled).cuda()

        # Train SOM
        som_builder.optimize(pc)
        som_node_np = som_builder.node.cpu().numpy().astype(np.float32)  # 3 x node_num
        
        # Save file as npz
        npz_file = os.path.join(output_root, f[0:-4]+'.npz')
        np.savez(npz_file, pc=pc_np, sn=sn_np, som_node=som_node_np) # sn = surface normal

if __name__ == "__main__":
    rows, cols = 8, 8
    if len(sys.argv) == 3:
        rows, cols = sys.argv[1], sys.argv[2]

    som_saver_catenary_arches('/home/jovyan/catenary_data_norm_sn_pcd', rows, cols, 0, '/home/jovyan/catenary_data_norm_sn_pcd/%dx%d'%(rows,cols))


    # if file[-3:] == 'txt':
    #     data = np.loadtxt(os.path.join(root, folder, file))
    #     pc_np = data[:, 0:3] # Nx3 array -> [[x0, y0, z0], ..., [xn, yn, zn]]
    #     sn_np = data[:, 3:6]
        
    #     pc_np_sampled = pc_np[np.random.choice(pc_np.shape[0], 4096, replace=False), :]
    #     pc = torch.from_numpy(pc_np_sampled.transpose().astype(np.float32)).cuda()  # 3xN tensor -> [[x0, ..., xn], [y0, ..., yn, [z0, ..., zn]]]
    #     som_builder.optimize(pc)
    #     som_node_np = som_builder.node.cpu().numpy().transpose().astype(np.float32)  # node_numx3

    #     npz_file = os.path.join(output_root, file[0:-4]+'.npz')
    #     np.savez(npz_file, pc=pc_np, sn=sn_np, som_node=som_node_np)

    # som_builder.optimize(pc)  # pc: 3xN float tensor
    # node = som_builder.node  # 3xM float tensor