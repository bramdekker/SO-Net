import os
import os.path
import numpy as np
import laspy
import torch

from util import som

def som_saver_catenary_arches(root, rows, cols, gpu_ids, output_root):
    som_builder = som.SOM(rows, cols, 3, gpu_ids)
    
    # Prepare the point clouds as a 3xN pytorch tensor (N arrays with 3 coordinates!)
    file_list = os.listdir(root)
    for j, f in enumerate(file_list):
        if os.path.isdir(os.path.join(root, f)):
            continue

        f_las = laspy.read(os.path.join(root, f))
        
        pc_np = np.vstack((f_las.x, f_las.y, f_las.z))
        pc_np = np.transpose(pc_np)
        pc_sampled = pc_np[:, np.random.choice(pc_np.shape[0], 524288, replace=False)] # sample_size x 3
        pc = torch.from_numpy(pc_sampled).cuda()
        som_builder.optimize(pc)
        som_node_np = som_builder.node.cpu().numpy().transpose().astype(np.float32)  # node_num x 3
        
        npz_file = os.path.join(output_root, f[0:-4]+'.npz')
        np.savez(npz_file, pc=pc_np, som_node=som_node_np) # sn = surface normal

if __name__ == "__main__":
    rows, cols = 8, 8
    som_saver_catenary_arches('/home/jovyan/catenary-data', rows, cols, 0, '/home/jovyan/catenary-data/%dx%d'%(rows,cols))