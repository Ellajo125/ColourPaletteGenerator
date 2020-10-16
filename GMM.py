import numpy as np
from upload_image import UploadedImage
import kmeans


def randomize_parameter(k_num, data):
    """Function to randomize the parameters of the EM Algorithm"""

    # Setting the initial means using random data points
    mu = np.random.randint(data.min(), data.max(),(k_num,data.shape[1]))

    # Setting the initial sigma based off of the assigned mu
    weights = np.random.rand(data.shape[0], k_num)
    weights = weights / np.sum(weights, axis=1)[np.newaxis].T
    alpha = np.sum(weights, axis=0) / data.shape[0]
    sigma = cal_covariance(weights, mu, data)

    return alpha, mu.astype(int), sigma


def cal_covariance(weights, mu, data):
    """Function to calculate the co-variance of a GMM"""

    sigma = np.zeros((data.shape[1], data.shape[1], mu.shape[0]))

    for k_index in range(0, mu.shape[0]):
        # For loop for each for centroid
        dist = data-mu[k_index,:]
        weight = weights[:,k_index]
        sigma[:, :, k_index] = ((weight[np.newaxis].T * dist).T @ dist) / np.sum(weights[:, k_index])

    return sigma


def mixture_components(alpha, x, mu, inv_sigma, norm_sigma):
    """Function to calculate the Gaussian Density"""
    dist = x-mu

    try:
        pk = np.exp(-1/2 * (dist @ inv_sigma @ dist.T)) / (((2*alpha)**(x.shape[0]/2)) * norm_sigma**(1/2))

    except FloatingPointError:
        print('dist')
        print(dist)
        print('Inside E')
        print((dist @ inv_sigma @ dist.T))
        print('alpha')
        print(alpha)
        print('norm sigma')
        print(norm_sigma)
        print('xshape')
        print(x.shape[0])
        print("inv_sigma")
        print(inv_sigma)
        breakpoint()

    return pk


def e_step(alpha, mu, sigma, data):
    """Function to perform the expectation value function for gmm"""
    weights = np.zeros((data.shape[0], mu.shape[0]))
    inv_sigma = np.zeros( sigma.shape )
    norm_sigma = np.zeros((sigma.shape[2]))

    for k_index in range(0, mu.shape[0]):
        # Calculating inverses in advance
        inv_sigma[:, :, k_index] = np.linalg.pinv(sigma[:, :, k_index])
        norm_sigma[k_index] = np.linalg.norm(sigma[:, :, k_index ])

    # First find density of each cluster
    for point in range(0, data.shape[0]):
        temp = np.zeros((mu.shape[0]))

        for k_index in range(0, mu.shape[0]):
            temp[k_index] = mixture_components(alpha[k_index], data[point, :], mu[k_index, :], inv_sigma[:, :, k_index], norm_sigma[k_index]) \
                                      * alpha[k_index]
        weights[point, :] = temp / np.sum(temp)

    return weights


def m_step(weights, data):
    """Function to perform the maximization value function"""

    alpha = np.sum(weights, axis=0) / data.shape[0]
    mu = (weights.T @ data) / (alpha*data.shape[0])[np.newaxis].T
    sigma = cal_covariance(weights, mu, data)

    return alpha, mu.astype(int), sigma


def convergence_check(alpha, mu, sigma, data):
    """Function to check the convergence of EM"""

    log_l = 0

    for point in range(0, data.shape[0]):
        inside_log = 0
        for k_index in range(0,mu.shape[0]):
            inside_log += alpha[k_index] * mixture_components(alpha[k_index], data[point, :], mu[k_index, :], sigma[:, :, k_index])
        log_l += np.log(inside_log)

    return log_l


def run_em(k_num, data, itter_max=25, tol=.001):
    """Function to run the EM algorithm """
    j = 0  # Variable to track the number of iterations
    diff = 10 # Variable to start the comparative difference
    old_logl = 10

    alpha, mu, sigma = randomize_parameter(k_num, data)
    print(j)

    while j < itter_max:
        print('a')
        weights = e_step(alpha, mu, sigma, data)
        print('b')
        alpha, mu, sigma = m_step(weights, data)
        print('c')
        # logl = convergence_check(alpha,mu,sigma, data)

        # diff = abs(old_logl-logl)
        # old_logl = logl

        j += 1
        print(j)
        print(mu)

    return mu


if __name__ == '__main__':
    image1 = UploadedImage('test7.jpg')
    px_1 = image1.img_pixels()
    k = run_em(5, px_1)
    print('k')
    print(k)