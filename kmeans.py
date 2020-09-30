import numpy as np
from random import randint
from upload_image import UploadedImage
import time


"""This module is for holding the functions to perform the kmeans algorithm. 
Data is to be added as a numpy matrix with each row being a separate training example """


def randomize_k(k_num, data):
    """Randomize the inital values of k to perform first itereation of kmeans"""
    row_index, col_index = data.shape
    k_int = np.zeros((k_num, col_index),dtype='uint16')

    for k_index in range(0, k_num):
        k_int[k_index, :] = data[randint(0, row_index-1), :]
    return k_int


def remove_k_value(k_num, k, row_index):
    k = np.delete(k, row_index, axis=0)
    k_num -= 1
    return k_num, k


def data_to_k_(k_num, k, data):
    """Calculating the distance to each data point and assigning it to a k value"""
    row_index, col_index = data.shape
    data_dis = np.zeros((row_index, k_num))

    "Calculating the distance to each node"
    for k_index in range(0, k_num):
        data_dis[:, k_index] = (np.sum(((data-k[k_index,:])**2), axis=1))

    """Determine which node is to each data point"""
    assigned_k = data_dis == data_dis.min(axis=1)[np.newaxis].T
    return assigned_k


def k_mod(k_num, assigned_k, data):
    """Add if k node has none assigned"""
    row_index, col_index = data.shape
    k = np.zeros((k_num, col_index), dtype='uint16')

    for k_index in range(0, k_num):
        num_data_points = np.sum(assigned_k[:, k_index], dtype='uint32')
        try:
            k_temp = assigned_k[:, k_index]
            k[k_index, :] = np.sum(data * k_temp[np.newaxis].T, axis=0) / num_data_points
        except ZeroDivisionError:
            k_num, k = remove_k_value(k_num, k, k_index)

    return k, k_num


def run_kmeans(k_num, data, itter=25):
    """Function to run the kmeans stuff"""
    #Add some functionality to go through multiple times in case of locaL minima
    k = randomize_k(k_num, data)
    k_num_int = k_num
    j = 0
    t0=[]
    t1=[]
    t2=[]

    while j <= itter:
        t0.append(time.process_time())
        assigned_k = data_to_k_(k_num, k, data)
        t1.append(time.process_time())
        k, k_num = k_mod(k_num, assigned_k, data)
        t2.append(time.process_time())
        j += 1
        print(j)

    if k_num_int == k_num:
        same_k = True
    else:
        same_k = False
    bench = np.array((t0,t1,t2))
    return k, same_k,  bench



if __name__ == '__main__':
    image1 = UploadedImage('test10.jpg')
    px_1 = image1.img_pixels()
    k, same_k, bench = run_kmeans(10, px_1)
    time_data_to_k = 0
    time_k_mod =0

    for index in range(0,bench.shape[0]):
        time_data_to_k += (bench[index, 1]-bench[index, 0])
        time_k_mod += (bench[index, 2]-bench[index, 1])
    print("time_data_to_k")
    print(time_data_to_k/bench.shape[0])
    print("time_k_mod")
    print(time_k_mod / bench.shape[0])
    print('k')
    print(k)