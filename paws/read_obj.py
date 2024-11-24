import open3d as o3d
import numpy as np
import math

def read_obj(path,Nx,Ny,Nz,padding_coef,base_sound_speed,base_density,base_alpha,obj_sound_speed,obj_density,obj_alpha):

    #input
    mesh = o3d.io.read_triangle_mesh(path)

    # fit to unit cube [-0.5,0.5]
    
    mesh.translate(-(mesh.get_max_bound() + mesh.get_min_bound())/2)
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
            center=[0,0,0])
    
    mesh.translate((0.5,0.5,0.5))
    
    # o3d.visualization.draw_geometries([mesh])

    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                                voxel_size=0.05)
    # o3d.visualization.draw_geometries([voxel_grid])
    
    #initialize field
    sound_speed_grid = np.ones((Nx, Ny, Nz), dtype=float) * base_sound_speed
    density_grid = np.ones((Nx, Ny, Nz), dtype=float) * base_density
    alpha_grid = np.ones((Nx, Ny, Nz), dtype=float) * base_alpha
    
    base_range = max(Nx,Ny)

    #further adjust mesh size based on padding_coef
    temp_mesh = o3d.geometry.TriangleMesh(mesh)
    temp_mesh.scale(1-padding_coef,center=[0,0,0])

    #voxelize the mesh
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(temp_mesh,voxel_size=2/base_range)

    #update the grid over lapping with the bound box
    

    #original update
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):

                index = np.array([[(x+0.5)/Nx - 0.5,0,(y+0.5)/Ny - 0.5,(z+0.5)/Nz - 0.5]]) 
                index = o3d.utility.Vector3dVector(index)

                if(voxel_grid.check_if_included(index)[0]):
                    sound_speed_grid[x][y][z] = obj_sound_speed
                    density_grid[x][y][z] = obj_density
                    alpha_grid[x][y][z] = obj_alpha

    return sound_speed_grid,density_grid,alpha_grid
    
    

# def read_obj_plan_b(path):
    
#     file_max_bound = []
#     file_min_bound = []

#     file_max_bound = [-math.inf]*3
#     file_min_bound = [math.inf]*3

#     mesh_list = []

#     # find max and min bound
#     for file in selected_file:

#         filename = dir_path + "/" + file

#         mesh = o3d.io.read_triangle_mesh(filename)
        
#         file_max_bound = np.max([mesh.get_max_bound(),file_max_bound],0)
#         file_min_bound = np.min([mesh.get_min_bound(),file_min_bound],0)
        
        
#     # transform the mesh into unit scale [-0.5,0.5]

#     for file in selected_file:

#         filename = dir_path + "/" + file

#         mesh = o3d.io.read_triangle_mesh(filename)
        
        
#         # resize the mesh into [-0.5,0.5] unit square

#         mesh.translate(-(file_max_bound + file_min_bound)/2)

#         mesh.scale(1 / (np.max(file_max_bound - file_min_bound)),center=[0,0,0])

#         mesh_list.append(mesh)

#         voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
#                                                                 voxel_size=0.05)
#         o3d.visualization.draw_geometries([voxel_grid])

#         #show all of them
#         o3d.visualization.draw_geometries(mesh_list)