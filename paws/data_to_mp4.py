import numpy as np
import cv2
import pickle
import h5py
import os
import datetime
import tqdm

from .utils import mu_law, inv_mu_law, analog_to_digital, digital_to_analog

def test_encode():

    path = "complicated_room1_save1.npy"
    width = 128
    height = 128

    tensor = load_tensor(path=path, width=width, height=height)
    # (T, W, H)

    video_path = "output.mp4"
    meta_path = "output.pkl"

    v_scales = encode_tensor_to_video(
        tensor=tensor, 
        video_path=video_path,
    )
    # v_mins: (T,)
    # v_maxs: (T,)

    ###
    meta = {
        "v_scales": v_scales
    }
    pickle.dump(meta, open(meta_path, "wb"))
    print("Write out meta to {}".format(meta_path))


def test_decode():

    video_path = "output.mp4"
    meta_path = "output.pkl"

    meta = pickle.load(open(meta_path, "rb"))
    v_scales = meta["v_scales"]

    tensor = decode_video_to_tensor(
        video_path=video_path, 
        v_scales=v_scales
    )
    # (T, W, H)

    ### (Optional) Evaluation - Compare with ground truth ###
    path = "complicated_room1_save1.npy"
    width = 128
    height = 128

    gt_tensor = load_tensor(path=path, width=width, height=height)

    error = np.mean(np.abs(tensor - gt_tensor))
    rel_error = error / np.mean(np.abs(gt_tensor))
    print("Absolute Error: {:6f}".format(error))
    print("Relative Error: {:6f}".format(rel_error))


def load_tensor(path, width, height):

    loaded = np.load(path, allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["p"]
    data = data.reshape((data.shape[0], width, height),order='F')

    return data


def encode_tensor_to_video(tensor, video_path, fps=100):

    frames_num, width, height = tensor.shape

    v_scales = np.max(np.abs(tensor), axis=(1, 2))
    
    tmp = v_scales[:, None, None]

    tensor /= tmp
    tensor = mu_law(tensor)
    tensor = analog_to_digital(tensor)

    out = cv2.VideoWriter(
        filename=video_path, 
        fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
        fps=fps, 
        frameSize=(width, height), 
        isColor=False
    )

    for n in range(frames_num):
        out.write(tensor[n])
        
    out.release()
    
    print("Write video to {}".format(video_path))

    return v_scales



def encode_hdf5_to_video(hdf5_path,
                         save_dir,
                         height,
                         width,
                         save_filename_prefix="",
                         sub_frame_max=100,
                         down_sample_ratio=10,
                         fps=100):

    with h5py.File(hdf5_path,"r") as hf:

        frames_num= hf["p"].shape[1]
        # sub_frame_max = 100   #current best
        # room_size = 256
        height = height
        width = width

        video_path = os.path.join(save_dir,save_filename_prefix + "_video.mp4")
        meta_path = os.path.join(save_dir,save_filename_prefix + "_meta.pkl")

        out = cv2.VideoWriter(
            filename=video_path, 
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=fps, 
            frameSize=(width, height), 
            isColor=False
        )

        v_scales = np.array([])

        for current_frame in tqdm.tqdm(range(0,frames_num,sub_frame_max)):

            slice_frame_n = min(sub_frame_max, frames_num-current_frame)
            data_slice = hf["p"][0,current_frame:current_frame + slice_frame_n].copy()

            pressure_dist = data_slice[0::down_sample_ratio]
            pressure_dist = pressure_dist.reshape((pressure_dist.shape[0], height,width),order='F')

            sub_v_scales = np.max(np.abs(pressure_dist), axis=(1, 2))
            tmp = sub_v_scales[:, None, None]
            pressure_dist /= tmp

            pressure_dist = mu_law(pressure_dist)

            pressure_dist = analog_to_digital(pressure_dist)

            for n in range(pressure_dist.shape[0]):
                out.write(pressure_dist[n])
                 
            v_scales = np.concatenate((v_scales,sub_v_scales),0)

            #free memory
            # del pressure_dist
            # gc.collect()

        out.release()
        print("Write out video to {}".format(video_path))

        # print(v_scales)

        meta = {
            "v_scales": v_scales
        }

        pickle.dump(meta, open(meta_path, "wb"))
        print("Write out meta to {}".format(meta_path))



def decode_video_to_tensor(video_path, v_scales):

    cap = cv2.VideoCapture(filename=video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tensor = []

    while True:

        status, x = cap.read()  # (W, H, C)

        if not status:
            break

        tensor.append(x[:, :, 0])

    tensor = np.stack(tensor, axis=0).astype(np.int32)
    # (T, W, H)

    tensor = digital_to_analog(tensor)
    tensor = inv_mu_law(tensor)
    tensor *= v_scales[:, None, None]

    return tensor


if __name__ == '__main__':

    test_encode()
    test_decode()