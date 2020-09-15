import numpy as np
from random import randint

"""This module is for holding the functions to perform the kmeans algorithm. 
Data is to be added as a numpy matrix with each row being a separate training example """


def randomize_k(k_num, data):
    """Randomize the inital values of k to perform first itereation of kmeans"""
    row_index, col_index = data.shape
    k_int = np.zeros((k_num, col_index))

    for k_index in range(0, k_num-1):
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
    for k_index in range(0, k_num-1):

        data_dis[:, k_index] = (np.sum(((data-k[k_index,:])**2), axis=1))**(1/2)

    """Determine which node is to each data point"""
    assigned_k = data_dis == data_dis.min(axis=1)[np.newaxis].T


    return assigned_k , data_dis


def k_mod(k_num, k, assigned_k, data):
    """Add if k node has none assigned"""
    k_old = k

    for k_index in range(0, k_num-1):
        num_data_points = sum(assigned_k[:, k_index])

        try:
            k_temp = assigned_k[:, k_index]
            k[k_index, :] = sum(data * k_temp[np.newaxis].T) / num_data_points
        except ZeroDivisionError:

            k_num, k = remove_k_value(k_num, k, k_index)

    err = sum(sum(((k-k_old)/2)**2)**(1/2))

    return k, err, k_num


def run_kmeans(k_num, data, itter=100, tol=1*10^(-4)):
    """Function to run the kmeans stuff"""
    k = randomize_k(k_num,data)
    k_num_int = k_num
    j = 0
    err = 1
    err_list = []

    while (j <= itter) and (err >= tol):
        assigned_k, data_dis = data_to_k_(k_num, k ,data)
        k, err, k_num = k_mod(k_num, k, assigned_k, data)
        err_list.append(err)
        j += 1

    if k_num_int == k_num:
        same_k = True
    else:
        same_k = False
    return k, err_list, same_k

