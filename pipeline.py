import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import datetime
import gc
import cv2
import math

from paws.room_generation import shoe_box_pipeline,polygon_pipeline

from paws.data_to_mp4 import encode_tensor_to_video, decode_video_to_tensor

from paws.model_to_grid import make_source_2d, class_grid_to_medium_grid, make_sensor_2d, simulation_2d, point_cloud_voxelization,sample_source_2d,make_new_source

from paws.utils import mu_law, analog_to_digital


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

    for _ in range(20):

        #define grid
        room_size = 256
        grid_size = 2e-2
        Nt = 50000
        dt = 1e-6
        class_id_range = [1,9]
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # room_type = "shoe_box"
        # cls_grid,valid_mask = shoe_box_pipeline(256,0.1,0.1,8,10,class_id_range,[1,2,6])

        room_type = "polygon"
        cls_grid,valid_mask = polygon_pipeline(256,0.1,10,[5,10],class_id_range,[1,2,6])

        


        for sample_id in range(5):


            # save path of project
            # base_dir = os.getcwd()
            # save_dir = os.path.join(base_dir,"data")

            save_dir = "D:\\PAWS-dataset\\" + room_type
            save_filename = room_type + "_" + current_time + "_" + str(sample_id)

            save_path = os.path.join(save_dir,save_filename)


            


            medium = class_grid_to_medium_grid(cls_grid,class_id_dict)
            sensor = make_sensor_2d(np.ones([room_size,room_size],dtype=bool))


            source_x,source_y = sample_source_2d(valid_mask,cls_grid)
            source = make_source_2d(source_x,source_y,2,room_size,room_size,5)
            print("source_x = ",source_x)
            print("source_y = ",source_y)


            sensor_data = simulation_2d(room_size,room_size,grid_size,grid_size,source,medium,sensor,Nt,dt)
            # sensor_data,h5_path = simulation_2d(room_size,room_size,grid_size,grid_size,source,medium,sensor,Nt,dt)

            # print("h5_path = ")
            # print(h5_path)

            # break





            # pressure_dist = sensor_data["p"].copy()
            #down sampling
            pressure_dist = sensor_data["p"][0::10]
            pressure_dist = pressure_dist.reshape((pressure_dist.shape[0], room_size, room_size))


            sub_n = 10
            sub_size = math.floor(pressure_dist.shape[0] / sub_n)

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
            temp_file_dir = "C:\\Users\\Administrator\\AppData\\Local\\Temp"
            file_list = os.listdir(temp_file_dir) 

            for file in file_list:
                if file.find("kwave") != -1:
                    os.remove(os.path.join(temp_file_dir,file))
            


#该function是为了节省压制数据时的数据开销
#for each sub part of simulation, eg. Nt = 100000 in total, sub_Nt = 10000 for each part
def sub_pipeline(room_size,grid_size,source,medium,sensor,Nt,dt,out:cv2.VideoWriter,up_sample_rate=10):

    #for each sub part of simulation, 1s in total, 0.1s per part

    #further divide data generation into smaller blocks
    #dt = 1e-6, sub_sub_Nt = 20000


    sub_sensor_data = simulation_2d(room_size,room_size,grid_size,grid_size,source,medium,sensor,Nt,dt)


    # pressure_dist = sensor_data["p"].copy()
    #up sampling
    pressure_dist = sub_sensor_data["p"][0::up_sample_rate]
    pressure_dist = pressure_dist.reshape((pressure_dist.shape[0], room_size, room_size))

    next_p0 = pressure_dist[pressure_dist.shape[0]-1].copy().transpose()

    next_p = pressure_dist[pressure_dist.shape[0]-1].reshape((room_size*room_size,1)).copy()

    # next_ux = sub_sensor_data["ux"][-1].reshape((1,room_size,room_size)).copy()
    # next_uy = sub_sensor_data["uy"][-1].reshape((1,room_size,room_size)).copy()

    next_ux = sub_sensor_data["ux"][-1].reshape((room_size*room_size,1)).copy()
    next_uy = sub_sensor_data["uy"][-1].reshape((room_size*room_size,1)).copy()

    
    #tensor为即将需要写入的数据，随后替换
    tensor = pressure_dist
    frames_num = tensor.shape[0]

    sub_v_scales = np.max(np.abs(tensor), axis=(1, 2))
    tmp = sub_v_scales[:, None, None]

    tensor /= tmp
    tensor = mu_law(tensor)
    tensor = analog_to_digital(tensor)

    for n in range(frames_num):
        out.write(tensor[n])

    

    return next_p0,next_p,next_ux,next_uy,sub_v_scales
            

def new_pipeline(
                #  room_size = 256,
                #  grid_size = 2e-2,
                #  Nt = 100000,          #total frame of data
                #  dt = 1e-6,
                #  sub_Nt_max = 20000,       #max frame for each sub part
                #  class_id_range = [1,9],
                #  scene_n = 1,
                #  sample_n = 1,
                #  room_type = "shoe_box" 

):

    #for each scene
    for _ in range(1):

        #define grid
        room_size = 256
        grid_size = 2e-2
        Nt = 40000          #total frame of data
        dt = 1e-6

        sub_Nt_max = 100       #max frame for each sub part
        # sub_sub_Nt = 20000   #frame for sub sub part

        class_id_range = [1,9]
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        room_type = "shoe_box"
        cls_grid,valid_mask = shoe_box_pipeline(256,0.1,0.1,8,10,class_id_range,[1,2,6])

        # room_type = "polygon"
        # cls_grid,valid_mask = polygon_pipeline(256,0.1,10,[5,10],class_id_range,[1,2,6])

        
        #for each sound source
        for sample_id in range(1):

            save_dir = "D:\\PAWS-dataset\\generation_test"
            save_filename = room_type + "_" + current_time + "_" + str(sample_id)
            save_path = os.path.join(save_dir,save_filename)

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


            #初始化视频生成，随后将多次模拟的数据依次write进VideoWriter
            fps = 100
            #save pressure distribution
            video_path = save_path + "_pressure.mp4"
            meta_path = save_path + "_pressure.pkl"

            height = room_size
            width = room_size

            # frames_num, width, height = tensor.shape
            out = cv2.VideoWriter(
                filename=video_path, 
                fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
                fps=fps, 
                frameSize=(height, width), 
                isColor=False
            )


            # time_tag = str(sub_part*0.1) + "_" + str((sub_part+1)*0.1)

            # temp_save_path = save_path + " " + time_tag

            # print(time_tag)
            # print("temp_save_path = ",temp_save_path)

            #for each sub part of simulation, eg. Nt = 100000 in total, sub_Nt = 10000 for each part

            
            completed_Nt = 0
            v_scales = np.array([])

            while completed_Nt < Nt:

                temp_Nt = min(sub_Nt_max, Nt-completed_Nt)

                completed_Nt += temp_Nt

                next_p0,next_p,next_ux,next_uy,sub_v_scales = sub_pipeline(room_size,grid_size,source,medium,sensor,temp_Nt,dt,out,10)

                print("________________")

                print(next_p0)

                print(next_p)

                print(next_ux)

                print(next_uy)

                print("________________")

                print(sub_v_scales)
                    
                v_scales = np.concatenate((v_scales,sub_v_scales),0)


                source = make_new_source(next_p0,next_p,next_ux,next_uy,room_size,room_size)


                #free memory
                # del sub_sensor_data
                gc.collect()

                #free temp file
                temp_file_dir = "C:\\Users\\Administrator\\AppData\\Local\\Temp"
                file_list = os.listdir(temp_file_dir) 

                for file in file_list:
                    if file.find("kwave") != -1:
                        os.remove(os.path.join(temp_file_dir,file))


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

            #更新完成后release
            out.release()
            
            print("Write video to {}".format(video_path))

    return





if __name__ == "__main__":


    old_pipeline()

    # new_pipeline()





