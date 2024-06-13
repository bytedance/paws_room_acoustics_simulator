import numpy as np
import cv2
import pickle
import h5py

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
    data = data.reshape((data.shape[0], width, height))

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
        frameSize=(height, width), 
        isColor=False
    )

    for n in range(frames_num):
        out.write(tensor[n])
        
    out.release()
    
    print("Write video to {}".format(video_path))

    return v_scales



# def encode_hdf5_to_video(hdf5_path,video_path, fps=100)



#     h5_path = "C:\\Users\\Administrator\\AppData\\Local\\Temp13-Jun-2024-03-17-45_kwave_input.h5"



#     with h5py.File(h5_path,"r") as f:
#         # for key in f.keys():
#         #     #print(f[key], key, f[key].name, f[key].value) # 因为这里有group对象它是没有value属性的,故会异常。另外字符串读出来是字节流，需要解码成字符串。
#         #     print(f[key], key, f[key].name) 
#         #     print("----------")


#         p_data = f["p"]

#         frames_num = p_data.shape[1]


#         #tensor为即将需要写入的数据，随后替换
#         tensor = pressure_dist
#         frames_num = tensor.shape[0]

#         sub_v_scales = np.max(np.abs(tensor), axis=(1, 2))
#         tmp = sub_v_scales[:, None, None]

#         tensor /= tmp
#         tensor = mu_law(tensor)
#         tensor = analog_to_digital(tensor)

#         for n in range(frames_num):
#             out.write(tensor[n])







def encode_tensor_to_video_new(tensor, video_path, fps=100):

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
        frameSize=(height, width), 
        isColor=False
    )

    for n in range(frames_num):
        out.write(tensor[n])
        
    out.release()
    
    print("Write video to {}".format(video_path))

    return v_scales





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