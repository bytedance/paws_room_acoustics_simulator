import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import datetime
import gc
import math
import argparse

from paws.room_generation import shoe_box_pipeline,polygon_pipeline

from paws.data_to_mp4 import encode_tensor_to_video, encode_hdf5_to_video

from paws.model_to_grid import make_source_2d, class_grid_to_medium_grid, make_sensor_2d, simulation_2d,sample_source_2d



class_id_dict = {0:(100,330,0.75),      #air
                 1:(1000,3700,0.75),      #wood
                 2:(1440,4000,0.75),     #concrete
                 3:(1000,330,0.75),       #frabric
                 4:(2400,5300,0.75),    #china
                 5:(1390,2200,0.75),     #plastic
                 6:(2560,3810,0.75),     #stone
                 7:(7700,5000,0.75),     #stell
                 8:(4000,5600,0.75),     #glass
                 9:(1000,2210,0.75),      #wax
                 
}


def pipeline(args):
    
    print(args)
    
    #define basic parameter
    Nx = args.Nx
    Ny = args.Ny
    Nz = args.Nz
    
    dx = args.dx
    dy = args.dy
    dz = args.dz
    
    Nt = args.Nt
    dt = args.dt
    
    scene_n = args.scene_n
    source_n = args.source_n
    
    down_sample_ratio = args.down_sample_ratio
    room_type = args.room_type
    save_dir = args.save_dir
    keep_temp = args.keep_temp
    
    #get base path
    if save_dir == None:
        save_dir = os.getcwd()
    
    #define path
    temp_file_dir = os.path.join(save_dir,"temp_hdf5")
    path_validation(temp_file_dir)

    # save_dir = "/home/tianming/PAWS-dataset/" + room_type
    data_dir = os.path.join(save_dir,"generated_data")
    path_validation(data_dir)
    
    
    #class lookup table
    class_id_range = [1,9]
    class_id_dict = {0:(100,330,0.75),      #air
                        1:(1000,3700,0.75),      #wood
                        2:(1440,4000,0.75),     #concrete
                        3:(1000,330,0.75),       #frabric
                        4:(2400,5300,0.75),    #china
                        5:(1390,2200,0.75),     #plastic
                        6:(2560,3810,0.75),     #stone
                        7:(7700,5000,0.75),     #stell
                        8:(4000,5600,0.75),     #glass
                        9:(1000,2210,0.75),      #wax           
                    }
    
    
    #for each scene
    for _ in range(scene_n):
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        #define room
        if room_type == "shoe_box":
            cls_grid,valid_mask = shoe_box_pipeline(Nx,Ny,0.1,0.1,8,10,class_id_range,[1,2,6])

        elif room_type == "polygon":
            cls_grid,valid_mask = polygon_pipeline(Nx,Ny,0.1,10,[5,10],class_id_range,[1,2,6])

        else:
            print("Invalid room type: ", room_type, ", using shoe_box instead")
            cls_grid,valid_mask = shoe_box_pipeline(Nx,Ny,0.1,0.1,8,10,class_id_range,[1,2,6])
        

        # show demo of sampled area
        plt.figure()
        plt.imshow(np.squeeze(cls_grid), aspect='equal', cmap='gray')
        plt.xlabel('x-position [m]')
        plt.ylabel('y-position [m]')
        plt.title('generated room')
        plt.savefig(os.path.join(data_dir,room_type + "_" + current_time + "_cls_id.png"))


        ##for each source
        for sample_id in range(source_n):

            save_filename_prefix = room_type + "_" + current_time + "_" + str(sample_id)
            # save_path = os.path.join(save_dir,save_filename)

            medium = class_grid_to_medium_grid(cls_grid,class_id_dict)
            sensor = make_sensor_2d(np.ones([Nx,Ny],dtype=bool))

            source_x,source_y = sample_source_2d(valid_mask,cls_grid)
            source = make_source_2d(source_x,source_y,2,Nx,Ny,5)
            print("source_x = ",source_x)
            print("source_y = ",source_y)

            input_filename = current_time+"_kwave_input.h5"
            output_filename = current_time+"_kwave_output.h5"

            #generate hdf5 file
            simulation_2d(Nx,Ny,dx,dy,source,medium,sensor,Nt,dt,
                          data_path=temp_file_dir,
                          output_filename=output_filename,
                          input_fileneme=input_filename,
                         )

            hdf5_input_path = os.path.join(temp_file_dir,input_filename)
            hdf5_output_path = os.path.join(temp_file_dir,output_filename)

            encode_hdf5_to_video(hdf5_output_path,data_dir,Nx,Ny,save_filename_prefix,down_sample_ratio)

            #optional: remove hdf5 file
            if not keep_temp:
                os.remove(hdf5_input_path)
                os.remove(hdf5_output_path)

    return



def old_pipeline():

    #for each scene
    for _ in range(1):

        #define grid
        room_size = 256
        grid_size = 2e-2
        Nt = 10000
        dt = 1e-6
        class_id_range = [1,9]
        
        #get base path
        base_path = os.getcwd()
        
        temp_file_dir = os.path.join(base_path,"temp_hdf5")
        # temp_file_dir = None
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        room_type = "shoe_box"
        cls_grid,valid_mask = shoe_box_pipeline(256,0.1,0.1,8,10,class_id_range,[1,2,6])

        # room_type = "polygon"
        # cls_grid,valid_mask = polygon_pipeline(256,0.1,10,[5,10],class_id_range,[1,2,6])

        
        ##for cach source
        for sample_id in range(1):


            # save path of project
            # base_dir = os.getcwd()
            # save_dir = os.path.join(base_dir,"data")

            save_dir = os.path.join(base_path,room_type)
            save_filename = room_type + "_" + current_time + "_" + str(sample_id)

            save_path = os.path.join(save_dir,save_filename)


            medium = class_grid_to_medium_grid(cls_grid,class_id_dict)
            sensor = make_sensor_2d(np.ones([room_size,room_size],dtype=bool))


            source_x,source_y = sample_source_2d(valid_mask,cls_grid)
            source = make_source_2d(source_x,source_y,2,room_size,room_size,5)
            print("source_x = ",source_x)
            print("source_y = ",source_y)



            sensor_data = simulation_2d(room_size,room_size,grid_size,grid_size,source,medium,sensor,Nt,dt,
                                        data_path=temp_file_dir,
                                        output_filename=current_time+"_kwave_output.h5",
                                        input_fileneme=current_time+"_kwave_input.h5",
                                        )
            
            # sensor_data,h5_path = simulation_2d(room_size,room_size,grid_size,grid_size,source,medium,sensor,Nt,dt)

            # print("h5_path = ")
            # print(h5_path)

            # break
            break

            # pressure_dist = sensor_data["p"].copy()
            #down sampling
            pressure_dist = sensor_data["p"][0::10]
            pressure_dist = pressure_dist.reshape((pressure_dist.shape[0], room_size, room_size))


            sub_n = 10
            sub_size = math.floor(pressure_dist.shape[0] / sub_n)

            ##slice the generated data
            for sub_id in range(sub_n):

                sub_save_path = save_path + "_" + str(sub_id)

                # show demo of sampled area
                plt.figure()
                plt.imshow(np.squeeze(cls_grid), aspect='equal', cmap='gray')
                plt.xlabel('x-position [m]')
                plt.ylabel('y-position [m]')
                plt.title('generated room')
                plt.savefig(sub_save_path+"_cls_id.png")


                #save pressure distribution
                video_path = sub_save_path + "_pressure.mp4"
                meta_path = sub_save_path +"_pressure.pkl"

                sub_data = pressure_dist[sub_size*sub_id : sub_size*(sub_id+1)-1].copy()


                v_scales = encode_tensor_to_video(
                    tensor=sub_data, 
                    video_path=video_path,
                )
                # v_mins: (T,)
                # v_maxs: (T,)
                ###

                # print(v_scales)

                meta = {
                    "v_scales": v_scales
                }
                pickle.dump(meta, open(meta_path, "wb"))
                print("Write out meta to {}".format(meta_path))

                #save medium data
                medium_save_pth = sub_save_path + "_medium.npy"
                medium_data = {"sound_speed":medium.sound_speed,
                                "density":medium.density}

                np.save(medium_save_pth,medium_data)


            #free memory
            del sensor_data,pressure_dist
            gc.collect()

            #free temp file

            # temp_file_dir = "/home/tianming/git_project/PAWS-Dataset/temp_hdf5"
            clear_temp(temp_file_dir)


def path_validation(path:str):
    
    if os.path.exists(path):
        print("path: ", path, "validated")
    else:
        print("path: " + path + " doesn't exist, try to create")
        os.mkdir(path)
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input the parameter for simulation')

    parser.add_argument("--Nx",type=int,
                        help='The number of grid along x-axis', default=256)
    parser.add_argument("--Ny",type=int,
                        help='The number of grid along y-axis', default=256)
    parser.add_argument("--Nz",type=int,
                        help='The number of grid along z-axis, set to 0 if the simulation is in 2D', default=0)

    parser.add_argument("--dx",type=float,
                        help='The size of grid along x-axis', default=2e-2)
    parser.add_argument("--dy",type=float,
                        help='The size of grid along y-axis', default=2e-2)
    parser.add_argument("--dz",type=float,
                        help='The size of grid along z-axis', default=2e-2)

    parser.add_argument("--Nt",type=int,
                        help='The total number of frame in simulation', default=1000)
    parser.add_argument("--dt",type=float,
                        help='The time interval between two frames in simulation', default=1e-6)

    parser.add_argument("--scene_n",type=int,
                        help='The number of scene used in simulation', default=1)
    parser.add_argument("--source_n",type=int,
                        help='The number of source sampled in each scene', default=1)
    
    parser.add_argument("--down_sample_ratio",type=int,
                        help='The down sample ratio of simulation result', default=10)
    parser.add_argument("--room_type",type=str,
                        help='The scene type of simulation, the valid inputs are \"shoe_box\" and \"polygon\"', default="shoe_box")
    parser.add_argument("--save_dir",type=str,
                        help='The location to save the simulation result', default=None)
    parser.add_argument("--keep_temp", action="store_true",
                    help='If set to True, the simulation will keep the temp hdf5 file')

    args = parser.parse_args()

    pipeline(args)
    





