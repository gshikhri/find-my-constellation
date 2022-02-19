import numpy as np
import os
import pickle
from scipy import ndimage
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import KDTree, distance
from scipy.optimize import curve_fit
from skimage import io
import pandas as pd


def get_first_sub_image(coord, first_frame, shape):
    cluster_pixels = first_frame[coord[0]-int(shape[0]/2): coord[0]+int(round(shape[0]/2) + 1), \
        coord[1]-int(shape[1]/2): coord[1]+int(round(shape[1]/2) + 1)]
    return cluster_pixels

def get_se_coord_intensity(coord, kinetics_stack):
    return kinetics_stack[:, coord[0], coord[1]]

def compare_images(image_list, rc, image_title_list):
    rows= rc[0]
    cols = rc[1]
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(image_list[i], cmap='gray')
        ax.set_title(image_title_list[i])
    plt.show()

def get_distance_list(se_cluster_list, fq_cluster_list):
    distance_list = []
    for se_cluster, fq_cluster in zip(se_cluster_list, fq_cluster_list):
        distance_list.append(distance.euclidean(se_cluster, fq_cluster ))
    return distance_list

def filter_hit_clusters(hit_se_clusters, hit_fq_clusters):
    filtered_se_clusters = []
    filtered_fq_clusters = []
    distance_list = get_distance_list(hit_se_clusters, hit_fq_clusters)
    mean = np.mean(distance_list)
    std = np.std(distance_list)
    for se_cluster, fq_cluster in zip(hit_se_clusters, hit_fq_clusters):
        if (mean - std <= distance.euclidean(se_cluster, fq_cluster) <= mean + std):
            filtered_se_clusters.append(se_cluster)
            filtered_fq_clusters.append(fq_cluster)
    return filtered_se_clusters, filtered_fq_clusters, distance_list

def get_exclusive_hits(mutual_hits, non_mutual_hits):
    se_cluster_in_non_mutual = set(i for i, j in non_mutual_hits)
    fq_cluster_in_non_mutual = set(j for i, j in non_mutual_hits)
    
    exclusive_hits = set((i, j) for i, j in mutual_hits 
                         if i not in se_cluster_in_non_mutual and j not in fq_cluster_in_non_mutual)
    return (exclusive_hits)

def do_constellation_mapping(se_cluster_pos, fq_cluster_pos):
    ##performing the classify hits algorithm, starting with making KDTree of se and aligned fq clusters
    se_cluster_tree = KDTree(se_cluster_pos)
    fq_cluster_tree = KDTree(fq_cluster_pos)
    
    #creating mapping sets
    se_to_fq= set()
    fq_to_se= set()
    
    for i, pt in enumerate(se_cluster_pos):
        dist, idx = fq_cluster_tree.query(pt)
        se_to_fq.add((i, idx))
        
    for i, pt in enumerate(fq_cluster_pos):
        dist, idx = se_cluster_tree.query(pt)
        fq_to_se.add((idx, i))
        
    mutual_hits = se_to_fq & fq_to_se
    non_mutual_hits = se_to_fq ^ fq_to_se
    
    return (mutual_hits, non_mutual_hits)

def make_image_from_coords(coords, image_size):
    image = np.zeros(shape=image_size, dtype=np.float32)
    for i in coords:
        if (0<i[0]<2048) and (0<i[1]<2048):
            image[int(i[0]), int(i[1])] = 1
    return(image)
    # return (ndimage.gaussian_filter(image, sigma))

def pad_to_size(M, size):
    assert len(size) == 2, 'Row and column sizes needed.'
    left_to_pad = size - np.array(M.shape)
    return np.pad(M, ((0, left_to_pad[0]), (0, left_to_pad[1])), mode='constant')

def get_target_tile_subimage(aligned_tile, aligned_rc, image_size, tile_physical_width, pad, channel):
    fastq_image_dir = 'tile_based_fastq_images'
    tile_fastq_image_name = '{}_{}_point_image_{}_um.tif'.format(channel, aligned_tile, 
                                                                 tile_physical_width)
    tile_fastq_sub_image = io.imread(os.path.join(fastq_image_dir, 
                                              tile_fastq_image_name))[aligned_rc[0] - pad:aligned_rc[0] + image_size[0] + pad, 
                                                                      aligned_rc[1] - pad:aligned_rc[1] + image_size[1] + pad]
    return tile_fastq_sub_image

se_cluster_pos = np.transpose(np.where(se_sub_image == 1)).tolist()
fq_cluster_pos = np.transpose(np.where(aligned_fq_sub_image == 1)).tolist()

mutual_hits, non_mutual_hits = do_constellation_mapping(se_cluster_pos, fq_cluster_pos)

exclusive_hits = get_exclusive_hits(mutual_hits, non_mutual_hits)

hit_se_clusters = [se_cluster_pos[i[0]] for i in exclusive_hits]
hit_fq_clusters = [fq_cluster_pos[i[1]] for i in exclusive_hits]

# hit_se_image = make_image_from_coords(hit_se_clusters, image_size=(sub_image_size, sub_image_size))
# hit_fq_image = make_image_from_coords(hit_fq_clusters, image_size=(sub_image_size, sub_image_size))

# compare_images([hit_se_image, hit_fq_image], rc=(1, 2), \
#     image_title_list=['filtered_se_image', 'filtered_fq_image'])


filtered_se_clusters, filtered_fq_clusters, distance_list = filter_hit_clusters(hit_se_clusters, hit_fq_clusters)

filtered_se_image = make_image_from_coords(filtered_se_clusters, image_size=(sub_image_size, sub_image_size))
filtered_fq_image = make_image_from_coords(filtered_fq_clusters, image_size=(sub_image_size, sub_image_size))

compare_images([filtered_se_image, filtered_fq_image], rc=(1, 2), \
    image_title_list=['filtered_se_image', 'filtered_fq_image'])

io.imsave('filtered_se_image_point.tif', filtered_se_image)
io.imsave('filtered_fq_image_point.tif', filtered_fq_image)

src = np.array(filtered_se_clusters)
dst = np.array(filtered_fq_clusters)

full_image_filtered_se = [] ## stores the filtered se_cluster coords
full_image_filtered_fq = [] ## stores the filtered fq_cluster coords
full_image_filtered_tile_fq = []

full_image_filtered_se.append(np.vstack((sub_image_size*row+src[:, 0], sub_image_size*col+src[:, 1])).T)

full_image_filtered_fq.append(np.vstack((sub_image_size*row+dst[:, 0], sub_image_size*col+dst[:, 1])).T)

full_image_filtered_tile_fq.append(np.vstack((aligned_rc[0] - pad + sub_im_aligned_rc[0] + dst[:, 0], 
                                                aligned_rc[1] - pad + sub_im_aligned_rc[1] + dst[:, 1])).T)
        
full_image_filtered_se = np.vstack(full_image_filtered_se)
full_image_filtered_fq = np.vstack(full_image_filtered_fq)
full_image_filtered_tile_fq = np.vstack(full_image_filtered_tile_fq)

dataset_pickle_file_name = os.path.join(fastq_image_dir, 
                                        "fq_read_dataset_{}".format(aligned_tile))
with open(dataset_pickle_file_name, 'rb') as read_file:
    dataset = pickle.load(read_file)

fq_cluster_pos_map = defaultdict()

fq_pos = np.array(dataset['target_fq_reads'])
scaled_fq_pos = np.array(dataset['target_scaled_fq_reads'])

# ## load fastq file for that tile
pos_seq_dir = 'tile_based_pos_seq_dict'
in_file_name = 'target_tile_{}_pos_seq_dict'.format(aligned_tile)
in_file_path = os.path.join(pos_seq_dir, in_file_name)
with open(in_file_path, 'rb') as read_file:
    target_cluster_pos_seq_dict = pickle.load(read_file)

for fq_cluster_in_tile, se_cluster, fq_cluster, se_sub_image_cluster, fq_sub_image_cluster in \
    zip(full_image_filtered_tile_fq, full_image_filtered_se, full_image_filtered_fq, src, dst):
    pos_in_fq = fq_pos[((scaled_fq_pos[:, 0] == fq_cluster_in_tile[0]) & 
                        (scaled_fq_pos[:, 1] == fq_cluster_in_tile[1]))][0]
    seq = target_cluster_pos_seq_dict[tuple(pos_in_fq)][5:]
    cluster_dict = dict()
    cluster_dict['fq_tile_image_pos'] = fq_cluster_in_tile
    cluster_dict['se_image_pos'] = se_cluster
    cluster_dict['fq_image_pos'] = fq_cluster
    cluster_dict['seq'] = seq
    cluster_dict['sub_image_se'] = se_sub_image_cluster
    cluster_dict['sub_image_fq'] = fq_sub_image_cluster
    fq_cluster_pos_map[tuple(pos_in_fq)] = cluster_dict

assign_out_dir = 'sub_image_assign_data'
if not os.path.exists(assign_out_dir):
    os.mkdir(assign_out_dir)


