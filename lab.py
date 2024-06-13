import h5py

import pandas as pd

import vaex




h5_path = "C:\\Users\\Administrator\\AppData\\Local\\Temp\\13-Jun-2024-03-17-45_kwave_input.h5"


df = vaex.open(h5_path)

print(f'Number of rows: {df.shape[0]:,}')
print(f'Number of columns: {df.shape[1]}')




import h5py

# 打开HDF5文件
file = h5py.File(h5_path, 'r')

# 获取需要读取的数据集
dataset = file['p']

# 设置读取的块大小
chunk_size = 1 

print(dataset.shape)

# 使用流式读取的方式读取数据
for i in range(0, dataset.shape[1], chunk_size):
    data = dataset[0][i:i+chunk_size]
    # 对读取的数据块进行处理
    print(data)

# 关闭文件
file.close()







# hdf5 = pd.HDFStore(h5_path , mode="r")

# print(hdf5.keys())

# print(list(hdf5.items()))




# with h5py.File(h5_path,"r") as f:
#     # for key in f.keys():
#     # 	 #print(f[key], key, f[key].name, f[key].value) # 因为这里有group对象它是没有value属性的,故会异常。另外字符串读出来是字节流，需要解码成字符串。
#     #     print(f[key], key, f[key].name) 
#     #     print(f[key].shape)
#     #     print("----------")

#     p_data = f["p"]

#     print(p_data) 

#     print(p_data.shape)

#     frames_num = p_data.shape[1]

#     print(frames_num)


#     sampled = p_data[0][0:1][:]
    
#     print(sampled)