import numpy as np
from random import randint
from upload_image import UploadedImage


""" This module is for holding the functions to perform the kmeans algorithm. 
Data is to be added as a numpy matrix with each row being a separate training example """


def randomize_k(k_num, data):
    """Randomize the initial values of k to perform first iteration of kmeans"""

    row_index, col_index = data.shape
    k_int = np.zeros((k_num, col_index), dtype='uint16')

    for k_index in range(0, k_num):
        k_int[k_index, :] = data[randint(0, row_index-1), :]
    return k_int


def remove_k_value(k_num, k, row_index):
    k = np.delete(k, row_index, axis=0)
    k_num -= 1
    return k_num, k


def fix_multiple_assignments(assigned_k):
    """ Function to randomly assign a data point to a k_value when assigned to more than one"""

    # Take the data that needs to
    to_fix_rows = np.where(np.sum(assigned_k, axis=1) > 1)[0]
    to_fix_assign_k = assigned_k[to_fix_rows, :]

    assigned_k[to_fix_rows, :] = np.zeros((1, assigned_k.shape[1]))

    combined = np.concatenate((to_fix_rows[np.newaxis].T, to_fix_assign_k), axis=1)
    binary = [0, 1]

    for k_index in range(1, combined.shape[1]):

        col = (combined[:, k_index])[np.newaxis].T
        probs = [1 - 1/(combined.shape[1]-k_index), 1/(combined.shape[1]-k_index)]
        selection = np.random.choice(binary, size=(col.shape[0], 1), p=probs)
        col = col * selection
        rows_fixed = (combined[np.where(col == 1), 0])[0]
        assigned_k[rows_fixed, k_index-1] = 1
        combined = np.delete(combined, np.where(col == 1), axis=0)

    return assigned_k


def data_to_k_(k_num, k, data):
    """Calculating the distance to each data point and assigning it to a k value"""

    data_dis = data_distance(k_num, k, data)
    assigned_k = data_dis == (data_dis.min(axis=1))[np.newaxis].T

    if np.sum(assigned_k) > assigned_k.shape[0]:
        # If two points are assigned to two different clusters
        assigned_k = fix_multiple_assignments(assigned_k)

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
    """Function to modify the k values based  on the data points assigned to them"""

    row_index, col_index = data.shape
    k = np.zeros((k_num, col_index), dtype='uint16')

    for k_index in range(0, k_num):
        num_data_points = np.sum(assigned_k[:, k_index], dtype='uint32')
        try:

            k[k_index, :] = np.sum(data * (assigned_k[:, k_index])[np.newaxis].T, axis=0) / num_data_points
        except ZeroDivisionError:
            # Removes a k value if it has no points assigned to it
            k_num, k = remove_k_value(k_num, k, k_index)

    return k, k_num


def single_k_means(k_num, data, itter):
    """Function to run the k-means with a single k_value"""
    k = randomize_k(k_num, data)
    j = 0

    while j <= itter:
        assigned_k = data_to_k_(k_num, k, data)
        k, k_num = k_mod(k_num, assigned_k, data)
        j += 1

    return k, assigned_k


def sub_cluster_kmeans(data, k_initial, assigned_k, initial_split, sub_split, itter):
    """Function to run sub-clustering of kmeans"""

    kmax = initial_split*sub_split
    points_per_k = np.zeros((kmax, 1))
    k = np.zeros((kmax, k_initial.shape[1]))

    for k_index in range(0, initial_split):
        sub_data = data[~((data * (assigned_k[:, [k_index]])) == 0).all(1)]
        k[sub_split*k_index:sub_split*(k_index+1), :], sub_assign = single_k_means(sub_split, sub_data, itter)
        points_per_k[sub_split*k_index:sub_split*(k_index+1)] = np.sum(sub_assign, axis=0, keepdims=True).T

    return k, points_per_k


def remove_close_ks(k, points_per_k, cul_value):
    """Removes k-values that are close together. This starts by comparing the largest size to the smallest"""

    sort = np.argsort
    index = 0
    cul_value2 = cul_value ** 2

    # First order the k-values on there weight
    k_sorted = np.reshape((k[sort(points_per_k[:], axis=0)]), k.shape)[::-1]
    points_per_sorted = np.reshape((points_per_k[sort(points_per_k[:], axis=0)])[::-1], points_per_k.shape)

    # This loop below merges similar points until none meet the criteria of cul percent. This starts with the largest
    # value. Everytime a point is absorbed into another, their values are averaged out
    while k_sorted.shape[0] > index:
        point = (k_sorted[[index], :])
        dist = np.sum((point - k_sorted)**2, axis=1)
        combined_index = np.where(dist[index:] <= cul_value2)[0]

        if np.sum(combined_index) != 0:
            combined_index += index
            weights = (points_per_k[combined_index, :]) * np.ones(k_sorted[combined_index, :].shape)

            # Create a new point taking the average of the found similar points and combinding the weights.
            k_sorted[index, :] = np.average(k_sorted[combined_index, :], weights=weights, axis=0)
            points_per_sorted[index, :] = np.sum(weights[:, 0])

            # Removed similar values from the matrix
            k_sorted = np.delete(k_sorted, combined_index[1:], axis=0)
            points_per_sorted = np.delete(points_per_sorted, combined_index[1:], axis=0)

        index += 1

    k_sorted = np.reshape((k_sorted[sort(points_per_sorted[:], axis=0)]), k_sorted.shape)[::-1]
    points_per_sorted = np.reshape((points_per_sorted[sort(points_per_sorted[:], axis=0)])[::-1],
                                   points_per_sorted.shape)

    return k_sorted.astype(int), points_per_sorted


def run_k_means(data, initial_split=4, sub_split=4, itter=15, itter2=25, cull_percent=.1):
    """Function to run the single kmeans algorithm. This will perform an initial split of the data and then perform
     k means clustering the to clusters found in the initial split."""

    # Running the initial split of the data
    k_initial, assigned_k = single_k_means(initial_split, data, itter)

    # Perform k-means on the split data
    k, points_per_k = sub_cluster_kmeans(data, k_initial, assigned_k, initial_split, sub_split, itter2)

    # Remove points that are too close
    cull_value = np.max(data) * cull_percent * data.shape[1]
    k, points_sort = remove_close_ks(k, points_per_k, cull_value)

    return k, points_sort


if __name__ == '__main__':
    image1 = UploadedImage('test2.png')
    px_1 = image1.img_pixels()
    kout, points = run_k_means(px_1)

    print('k')
    print(kout)
    print('points')
    print(points)
