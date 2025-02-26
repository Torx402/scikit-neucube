import torch
import numpy as np

def small_world_connectivity(dist, c, l):
    """
    Calculates a small-world network connectivity matrix based on the given distance matrix.

    Args:
        dist (torch.Tensor): The distance matrix representing pairwise distances between nodes.
        c (float): Maximum connection probability
        l (float): Small world connection radius

    Returns:
        torch.Tensor: Connectivity matrix.

    """

    # Normalize the distance matrix
    dist_norm = (dist - torch.min(dist, dim=1).values[:, None]) / (torch.max(dist, dim=1).values[:, None] - torch.min(dist, dim=1).values[:, None])

    # Calculate the connection probability matrix
    conn_prob = c * torch.exp(-(dist_norm / l) ** 2)

    # Create the input connectivity matrix by selecting connections based on probability
    input_conn = torch.where(torch.rand_like(conn_prob) < conn_prob, conn_prob, torch.zeros_like(conn_prob)).fill_diagonal_(0)

    return input_conn

def SWC(dist, swr=0.15, input_n=False):
    """
    Calculates a small-world network connectivity matrix based on the given distance matrix.
    This implementation is based on https://doi.org/10.1007/s11063-020-10322-8
    
    Args:
        dist (torch.Tensor): The distance matrix representing pairwise distances between nodes.
        swr (float): Small world connection radius.
        input_n (flag): A flag to determine if the distance matrix being passed consists of input neurons (eeg coordinates) or not.

    Returns:
        torch.Tensor: Connectivity matrix.

    """

    # Normalize the distance matrix
    # the minimum of the entire matrix is used here, otherwise, a distance of 1 would mean one thing for one row, and another for different row.
    dist_min = torch.min(dist, dim=1).values[:, None].min()
    dist_max = torch.max(dist, dim=1).values[:, None].max()
    
    
    dist_norm = (dist - dist_min) / (dist_max - dist_min)

    # extracting matrix with connection(i, i) or connection(j, j) set to 0, in the case of an N x N matrix, returns a matrix with 0 along the diagonal
    # and 1 elsewhere
    conn_mat_non_self = torch.where(dist_norm == 0., torch.zeros_like(dist_norm), torch.ones_like(dist_norm))
    
    # set connection flag to 0 where distance is greater than swr, otherwise, sample from the matrix consisting of 0 connections from a neuron to itself
    # this creates a symmetric matrix in case the connection matrix is N x N  
    conn_mat = torch.where(dist_norm > swr, torch.zeros_like(dist_norm), conn_mat_non_self)

    if input_n:
        
        w_mat = torch.where(conn_mat == 1., torch.rand_like(conn_mat) / dist, torch.zeros_like(conn_mat))
        
    else:
        
        # for the connectivity matrix of the reservoir, a connection is made if the distance between two neurons, d(i, j) < swr, 
        # however this yields a matrix that not only has connections for all neuron pairs (i, j) that meet this requirement
        # but also for the pairs (j, i) yielding a symmetric matrix with bidirectional connections. 
        # as per the implementation presented in the paper above, one of those connections must be eliminated (set to 0) using a 50-50 chance,
        # splitting the matrix into an upper-triangular and a lower-triangular pair means that the upper-triangular matrix contains all (i, j), 
        # while the lower-triangular one has all (j, i) connections. 
        ut_conn_mat = torch.triu(conn_mat)
        lt_conn_mat = ut_conn_mat.T
        
        # generate upper-triangular and lower-triangular probability matrices to be used to eliminate bidirectional connections
        ut_rand = torch.where(ut_conn_mat == 1., torch.rand_like(ut_conn_mat), torch.zeros_like(ut_conn_mat))
        lt_rand = torch.where(lt_conn_mat == 1., (1-ut_rand.T), torch.zeros_like(lt_conn_mat))
        
        # consolidate lower and upper triangular probability matrices into one probability matrix
        m_rand = ut_rand + lt_rand
        
        # eliminate bidirectional connections based on the probability matrix
        conn_mat = torch.where(m_rand > 0.5, torch.ones_like(m_rand), torch.zeros_like(m_rand))

        w_mat = torch.where(conn_mat == 1., torch.rand_like(conn_mat) / dist, torch.zeros_like(conn_mat))
    
    return w_mat

def small_world_connectivity_alt(dist, c, l):
    """
    Calculates a small-world network connectivity matrix based on the given distance matrix.
    The differences between this function and the original (without the _test suffix) are:
    
    * The way maximum and minimum distances are calculated. 
    * Using an alternative approach to eliminating self connections (a connection from a neuron i to itself) 
    without using the .fill_diagonal__() method.

    Args:
        dist (torch.Tensor): The distance matrix representing pairwise distances between nodes.
        c (float): Maximum connection probability
        l (float): Small world connection radius

    Returns:
        torch.Tensor: Connectivity matrix.

    """

    # Calculate min and max distances
    dist_min = torch.min(dist, dim=1).values[:, None].min()
    dist_max = torch.max(dist, dim=1).values[:, None].max()
    
    # Normalize the distance matrix
    dist_norm = (dist - dist_min) / (dist_max - dist_min)

    # Calculate the connection probability matrix
    conn_prob = c * torch.exp(-(dist_norm / l) ** 2)

    # this approach ensures no self connections occur without using 
    # the .fill_diagonal_(0) method because it only works 
    # if the connection matrix is a N x N matrix, otherwise
    # it cannot be guaranteed that self connections are eliminated
    # especially in the case of input neurons
    conn_mat_non_self = torch.where(dist_norm == 0., torch.zeros_like(dist_norm), torch.ones_like(dist_norm))
    
    conn_prob_non_self = torch.where(conn_mat_non_self == 1., conn_prob, torch.zeros_like(conn_mat_non_self))

    # Create the input connectivity matrix by selecting connections based on probability
    input_conn = torch.where(torch.rand_like(conn_prob) < conn_prob_non_self, conn_prob_non_self, torch.zeros_like(conn_prob))

    return input_conn