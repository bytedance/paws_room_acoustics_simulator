import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DG
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.colormap import get_color_map
from kwave.utils.signals import tone_burst

import random
# import open3d.core as o3c

import os
import math

def class_grid_to_medium_grid(cls_grid:np.ndarray,lookup_dict:dict,base_sound_speed=343,base_density=100,base_alpha_coef=0.75):

    Nx,Ny = cls_grid.shape

    #define the medium
    sound_speed_grid = np.ones((Nx, Ny), dtype=float) * base_sound_speed
    density_grid = np.ones((Nx, Ny), dtype=float) * base_density
    alpha_grid = np.ones((Nx, Ny), dtype=float) * base_alpha_coef
    
    for x in range(Nx):
        for y in range(Ny):
            
            #if not defined
            # print(cls_grid[x][y])
            if cls_grid[x][y] == 0:
                continue
               
            density,sound_speed,alpha = lookup_dict[cls_grid[x][y]]

            sound_speed_grid[x][y] = sound_speed
            density_grid[x][y] = density
            alpha_grid[x][y] = alpha
            
    medium = kWaveMedium(sound_speed=sound_speed_grid, density=density_grid, alpha_coeff=alpha_grid, alpha_power=1.5,absorbing=True,stokes=True)
    
    return medium

def sample_source_2d(valid_mask,grid_data,max_attempt = 100,min_gap=3):
    x_range = valid_mask.shape[0]
    y_range = valid_mask.shape[0]

    finished = False
    while max_attempt > 0 and finished == False:

        source_x = random.randint(0,x_range-1)
        source_y = random.randint(0,y_range-1)

        if valid_mask[source_x][source_y] == 1 :

            finished = True
            for x in range(source_x-min_gap,source_x+min_gap):
                for y in range(source_y-min_gap,source_y+min_gap):
                    if grid_data[x][y] != 0:
                        finished = False

        max_attempt -= 1
        
    return source_x,source_y

def make_source_2d(x_pos:int,y_pos:int,radius:int,Nx:int,Ny:int,magnitude:int):

    #initial source
    source = kSource()
    p0 = np.zeros((Nx, Ny), dtype=float)

    for x in range(x_pos-radius-1,x_pos+radius+1):
        for y in range(y_pos-radius-1,y_pos+radius+1):
            if (x+0.5-x_pos)**2 + (y+0.5-y_pos)**2 < radius:
                p0[x,y] = magnitude

    source.p0 = p0

    return source


def make_new_source(p0,p,ux,uy,Nx:int,Ny:int):
    #initial source
    source = kSource()

    # source.p0 = p0

    source.p = np.array(p)
    source.p_mask =np.ones((Nx, Ny), dtype=bool)

    return source


def make_sensor_2d(sensor_mask):
    sensor = kSensor()
    sensor.mask = sensor_mask
    sensor.record = ["p","u"]

    return sensor

def point_cloud_voxelization(vertices_list,class_id_list,grid_range):

    # # Create a point cloud from python list.
    # pcd = o3d.t.geometry.PointCloud(vertices_list)
    
    #构建一个grid储存每一个grid材质的class id
    class_grid = np.zeros((grid_range, grid_range, grid_range), dtype=int)
    # id_2_cls = get_id_2_cls_dict(json_path)

    for i in range(vertices_list.shape[0]):
        
        #将点从[-0.5,0.5]投影到[0,base_range-1]
        # print(temp_point)
        temp_point = vertices_list[i]
        temp_point = (temp_point + 0.5) * math.floor(grid_range-1)
        
        #找到与点最近的grid
        [grid_x,grid_y,grid_z] = np.floor(temp_point)

        class_grid[int(grid_x)][int(grid_y)][int(grid_z)] = class_id_list[i]

    return class_grid


# def mesh_voxelization():

#     return


def simulation_2d(Nx,Ny,dx,dy,source,medium,sensor,
                  Nt:float=None,
                  dt:float=None,
                  data_path=None,
                  output_filename=None,
                  input_fileneme=None,
                  return_tensor = True,
                  ):

    #initial grid
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])


    #define Nt dt based on medium sound speed
    kgrid.makeTime(medium.sound_speed)

    if Nt != None:
        kgrid.Nt = Nt

    if dt != None:
        kgrid.dt = dt

    #simulation
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)

    simulation_options = SimulationOptions(
        save_to_disk=True,
        data_cast='single',
        data_path=data_path,
        output_filename=output_filename,
        input_filename=input_fileneme
    )


    print(simulation_options)

    if return_tensor:
        sim_data = kspace_first_order_2d_gpu(kgrid, source, sensor, medium, simulation_options, execution_options)
        return sim_data
    
    else:
        kspace_first_order_2d_gpu(kgrid, source, sensor, medium, simulation_options, execution_options)
        return




def simulation_3d(Nx,Ny,Nz,dx,dy,dz,source,medium,sensor,
                  Nt:float=None,
                  dt:float=None,
                  data_path=None,
                  output_filename=None,
                  input_fileneme=None,
                  return_tensor = True,
                  ):

    #initial grid
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])


    #define Nt dt based on medium sound speed
    kgrid.makeTime(medium.sound_speed)

    if Nt != None:
        kgrid.Nt = Nt

    if dt != None:
        kgrid.dt = dt

    #simulation
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)

    simulation_options = SimulationOptions(
        save_to_disk=True,
        data_cast='single',
        data_path=data_path,
        output_filename=output_filename,
        input_filename=input_fileneme
    )


    print(simulation_options)

    if return_tensor:
            
        sim_data = kspaceFirstOrder3DG(kgrid, source, sensor, medium, simulation_options, execution_options)
        return sim_data
    
    else:
        kspaceFirstOrder3DG(kgrid, source, sensor, medium, simulation_options, execution_options)
        return