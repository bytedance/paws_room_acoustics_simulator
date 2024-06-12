import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import datetime
import gc

from paws.room_generation import shoe_box_pipeline,polygon_pipeline

from paws.data_to_mp4 import encode_tensor_to_video, decode_video_to_tensor

from paws.model_to_grid import make_source_2d, class_grid_to_medium_grid, make_sensor_2d, simulation_2d, point_cloud_voxelization,sample_source_2d




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


def old_pipeline():

    for _ in range(1):

        #define grid
        room_size = 256
        grid_size = 2e-2
        Nt = 100000
        dt = 1e-6
        class_id_range = [1,9]
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # room_type = "shoe_box"
        # cls_grid,valid_mask = shoe_box_pipeline(256,0.1,0.1,8,10,class_id_range,[1,2,6])

        room_type = "polygon"
        cls_grid,valid_mask = polygon_pipeline(256,0.1,10,[5,10],class_id_range,[1,2,6])

        save_dir = "D:\PAWS-dataset"
        save_filename = room_type + "_" + current_time + "_" + str(sample_id)

        save_path = os.path.join(save_dir,save_filename)


        for sample_id in range(1):


            # save path of project
            # base_dir = os.getcwd()
            # save_dir = os.path.join(base_dir,"data")

            


            # show demo of sampled area
            plt.figure()
            plt.imshow(np.squeeze(cls_grid), aspect='equal', cmap='gray')
            plt.xlabel('x-position [m]')
            plt.ylabel('y-position [m]')
            plt.title('generated room')
            plt.savefig(save_path+"_cls_id.png")



            medium = class_grid_to_medium_grid(cls_grid,class_id_dict)
            sensor = make_sensor_2d(np.ones([room_size,room_size],dtype=bool))


            source_x,source_y = sample_source_2d(valid_mask,cls_grid)
            source = make_source_2d(source_x,source_y,2,room_size,room_size,5)
            print("source_x = ",source_x)
            print("source_y = ",source_y)


            sensor_data = simulation_2d(room_size,room_size,grid_size,grid_size,source,medium,sensor,Nt,dt)


            # save path of project
            base_dir = os.getcwd()
            save_dir = os.path.join(base_dir,"data")


            # pressure_dist = sensor_data["p"].copy()
            #up sampling
            pressure_dist = sensor_data["p"][0::10]
            pressure_dist = pressure_dist.reshape((pressure_dist.shape[0], room_size, room_size))

            #save pressure distribution
            video_path = save_path + "_pressure.mp4"
            meta_path = save_path + "_pressure.pkl"


            v_scales = encode_tensor_to_video(
                tensor=pressure_dist, 
                video_path=video_path,
            )
            # v_mins: (T,)
            # v_maxs: (T,)
            ###

            print(v_scales)

            meta = {
                "v_scales": v_scales
            }
            pickle.dump(meta, open(meta_path, "wb"))
            print("Write out meta to {}".format(meta_path))

            #save medium data
            medium_save_pth = save_path + "_medium.npy"
            medium_data = {"sound_speed":medium.sound_speed,
                            "density":medium.density}

            np.save(medium_save_pth,medium_data)


            #free memory
            del sensor_data,pressure_dist
            gc.collect()

            #free temp file
            temp_file_dir = "C:\\Users\\Administrator\\AppData\\Local\\Temp"
            file_list = os.listdir(temp_file_dir) 

            for file in file_list:
                if file.find("kwave") != -1:
                    os.remove(os.path.join(temp_file_dir,file))
            

def new_pipeline():

    for _ in range(1):

        #define grid
        room_size = 256
        grid_size = 2e-2
        Nt = 100000
        dt = 1e-6

        

        class_id_range = [1,9]
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # room_type = "shoe_box"
        # cls_grid,valid_mask = shoe_box_pipeline(256,0.1,0.1,8,10,class_id_range,[1,2,6])

        room_type = "polygon"
        cls_grid,valid_mask = polygon_pipeline(256,0.1,10,[5,10],class_id_range,[1,2,6])

        save_dir = "D:\PAWS-dataset"
        save_filename = room_type + "_" + current_time + "_" + str(sample_id)

        save_path = os.path.join(save_dir,save_filename)


        for sample_id in range(1):


            # save path of project
            # base_dir = os.getcwd()
            # save_dir = os.path.join(base_dir,"data")

            


            # show demo of sampled area
            plt.figure()
            plt.imshow(np.squeeze(cls_grid), aspect='equal', cmap='gray')
            plt.xlabel('x-position [m]')
            plt.ylabel('y-position [m]')
            plt.title('generated room')
            plt.savefig(save_path+"_cls_id.png")



            medium = class_grid_to_medium_grid(cls_grid,class_id_dict)
            sensor = make_sensor_2d(np.ones([room_size,room_size],dtype=bool))


            source_x,source_y = sample_source_2d(valid_mask,cls_grid)
            source = make_source_2d(source_x,source_y,2,room_size,room_size,5)
            print("source_x = ",source_x)
            print("source_y = ",source_y)


            sensor_data = simulation_2d(room_size,room_size,grid_size,grid_size,source,medium,sensor,Nt,dt)


            # save path of project
            base_dir = os.getcwd()
            save_dir = os.path.join(base_dir,"data")


            # pressure_dist = sensor_data["p"].copy()
            #up sampling
            pressure_dist = sensor_data["p"][0::10]
            pressure_dist = pressure_dist.reshape((pressure_dist.shape[0], room_size, room_size))

            #save pressure distribution
            video_path = save_path + "_pressure.mp4"
            meta_path = save_path + "_pressure.pkl"


            v_scales = encode_tensor_to_video(
                tensor=pressure_dist, 
                video_path=video_path,
            )
            # v_mins: (T,)
            # v_maxs: (T,)
            ###

            print(v_scales)

            meta = {
                "v_scales": v_scales
            }
            pickle.dump(meta, open(meta_path, "wb"))
            print("Write out meta to {}".format(meta_path))

            #save medium data
            medium_save_pth = save_path + "_medium.npy"
            medium_data = {"sound_speed":medium.sound_speed,
                            "density":medium.density}

            np.save(medium_save_pth,medium_data)


            #free memory
            del sensor_data,pressure_dist
            gc.collect()

            #free temp file
            temp_file_dir = "C:\\Users\\Administrator\\AppData\\Local\\Temp"
            file_list = os.listdir(temp_file_dir) 

            for file in file_list:
                if file.find("kwave") != -1:
                    os.remove(os.path.join(temp_file_dir,file))
            



    return







if __name__ == "__main__":




    new_pipeline()





