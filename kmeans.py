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

    data_dis = data_distance(k_num, k, data)
    """Determine which node is to each data point"""
    assigned_k = data_dis == data_dis.min(axis=1)[np.newaxis].T

    return assigned_k


def data_distance(k_num, k, data):
    """Calculating the distance of each point to a node"""
    row_index, col_index = data.shape
    data_dis = np.zeros((row_index, k_num))

    "Calculating the distance to each node"
    for k_index in range(0, k_num):
        data_dis[:, k_index] = (np.sum(((data - k[k_index, :]) ** 2), axis=1))**(1/2)

    return data_dis

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
    # k_check = k.shape[0]
    # k = np.unique(k, axis=0)
    # while k.shape[0] != k_check

    return k, k_num

def find_silCoe(k_num, assigned_k, k, data):
    """Determine the Silhouette Coefficiect for the number of k-values """
    invert_assigned_k = np.zeros(assigned_k.shape)
    data_dis = data_distance(k_num,k, data)
    x = assigned_k*data_dis
    x = x.max(1)

    "Using the inverted logical boolean of assigned_k to remove the max value"
    invert_assigned_k = invert_assigned_k == assigned_k

    "Removing the assigned k-value from the equation "
    data_dis_2 = ((data_dis+1) * invert_assigned_k)-1
    data_dis_2_invert = 1/data_dis_2
    y = 1 / np.max(data_dis_2_invert,1)
    print(min(y))
    print(' ')
    print(np.average(y))
    print(np.average(x))
    print(' ')
    silCoe = (np.average(y)-np.average(x))/max(np.average(y),np.average(x))

    return silCoe

def single_kmeans(k_num, data, itter):
    """Function to run the k-means with a single k_value"""
    #Add some functionality to go through multiple times in case of locaL minima
    k = randomize_k(k_num, data)
    j = 0

    while j <= itter:
        assigned_k = data_to_k_(k_num, k, data)
        k, k_num = k_mod(k_num, assigned_k, data)
        j += 1

    silCoe = find_silCoe(k_num, assigned_k, k, data)
    return k, silCoe

def run_kmeans(data, max_knum=4, redo=5, itter=100):
    """Function to run the single kmeans algorithm """

    silCoe = -1  # Coefficent to track the Silohouette Coeffiecent

    for k_num in range(3, max_knum):
        j=0

        while j < redo:
            k_new, silCoe_new = single_kmeans(k_num, data, itter)
            print(silCoe_new)
            if silCoe_new > silCoe:
                k = k_new
                silCoe = silCoe_new
            j +=1

    return k, silCoe


if __name__ == '__main__':
    image1 = UploadedImage('test2.png')
    px_1 = image1.img_pixels()
    k, silCoe = run_kmeans(px_1)

    print('k')
    print(k)
    print('silCoe')
    print(silCoe)