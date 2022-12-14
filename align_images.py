#!/usr/bin/env python
import argparse
import datetime
import json
import os
import time

import cv2 as cv
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from utils.argparse import get_args_parser


def get_matching_point_ids(img1_points_2d, img2_points_2d):
    img1_observed_points_2d = img1_points_2d[img1_points_2d[:, 2] != -1]
    img2_observed_points_2d = img2_points_2d[img2_points_2d[:, 2] != -1]

    # remove duplicate point ids
    img1_observed_point_unique_ids, img1_observed_point_unique_counts = np.unique(img1_observed_points_2d[:, 2], return_counts=True)
    img1_observed_point_non_duplicates = img1_observed_point_unique_ids[img1_observed_point_unique_counts == 1]
    img1_observed_points_mask = np.isin(img1_observed_points_2d[:, 2], img1_observed_point_non_duplicates)
    img1_observed_points_2d = img1_observed_points_2d[img1_observed_points_mask]

    img2_observed_point_unique_ids, img2_observed_point_unique_counts = np.unique(img2_observed_points_2d[:, 2], return_counts=True)
    img2_observed_point_non_duplicates = img2_observed_point_unique_ids[img2_observed_point_unique_counts == 1]
    img2_observed_points_mask = np.isin(img2_observed_points_2d[:, 2], img2_observed_point_non_duplicates)
    img2_observed_points_2d = img2_observed_points_2d[img2_observed_points_mask]

    img1_observed_point_ids = set(img1_observed_points_2d[:, 2])
    img2_observed_point_ids = set(img2_observed_points_2d[:, 2])

    matched_observed_point_ids = img1_observed_point_ids.intersection(img2_observed_point_ids)
    matched_observed_point_ids = list(matched_observed_point_ids)
    return matched_observed_point_ids

def project(src_pts, H):
    src_pts[:, 2] = 1
    projected_points = src_pts.dot(H)
    projected_points = projected_points / projected_points[:, 2].reshape(-1, 1)
    return projected_points

def run_ransac(pts_src, pts_dst, num_iterations, verbose=False):
    max_inliers = 0
    epsilon = 1

    for _ in range(num_iterations):
        random_indices = np.random.choice(np.arange(len(pts_src)), 4, replace=False)
        random_pts_src = pts_src[random_indices] 
        random_pts_dst = pts_dst[random_indices] 

        H, status = cv.findHomography(random_pts_src[:, :2], random_pts_dst[:, :2], 0)
        H = H.T
        projected_points = project(pts_src.copy(), H)
        distances = (((pts_dst[:, :2] - projected_points[:, :2])**2).sum(axis=1)**0.5)
        num_inliers = (distances < epsilon).sum()

        # Keep largest set of inliers
        if num_inliers > max_inliers:
            if verbose:
                print('max_inliers', max_inliers)
            max_inliers = num_inliers
            inlier_cordinates_for_maximum = np.where(distances < epsilon)
    return inlier_cordinates_for_maximum


def main(args):
    data_dir = f'data/{args.dataset}/images/'
    depth_data_dir = f'data/{args.dataset}/depths/'

    

    

    # output_dir = os.path.join('outputs', args.dataset)
    # os.makedirs(output_dir, exist_ok=True)

    with open(f'data/{args.dataset}.json', 'r') as f:
        id_to_timestamp = json.load(f)

    image_id_to_timetaken = {}
    for image_id in id_to_timestamp:
        s = id_to_timestamp[image_id]['datetaken']
        unix_time = time.mktime(datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
        image_id_to_timetaken[image_id] = unix_time
    del id_to_timestamp


    # get intrisics
    with open(f'data/{args.dataset}/sparse/{args.cluster_id}/cameras.txt') as f:
        lines = f.readlines()
        
    camera_id_to_data = {}

    for line in lines:
        if '#' in line:
            continue
        line = line.replace('\n', '')
        CAMERA_ID, MODEL, WIDTH, HEIGHT, *PARAMS = line.split()
        WIDTH = int(WIDTH)
        HEIGHT = int(HEIGHT)

        camera_id_to_data[CAMERA_ID] = {'model': MODEL, 'width': WIDTH, 'height': HEIGHT, 'params': PARAMS}



    with open(f'data/{args.dataset}/sparse/{args.cluster_id}/images.txt') as f:
        lines = f.readlines()  

    img_filename_to_metadata = {}
    img_filename_to_2d_points = {}

    count = 0
    point_2d_line = False
    for line in lines:
        if '#' in line:
            continue
        line = line.replace('\n', '')
        
        if point_2d_line:
            assert len(line.split()) % 3 == 0
            points_2d = np.array(line.split()).reshape(-1, 3).astype(float)
            img_filename_to_2d_points[img_filename] = points_2d
            point_2d_line = False
        else:
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, img_filename = line.split()
            qw, qx, qy, qz, tx, ty, tz = float(qw), float(qx), float(qy), float(qz), float(tx), float(ty), float(tz)
            img_filename_to_metadata[img_filename] = [image_id, qw, qx, qy, qz, tx, ty, tz, camera_id]
            point_2d_line = True
        
        count += 1



    # with open(f'data/{args.dataset}/sparse/{args.cluster_id}/points3D.txt') as f:
    #     lines = f.readlines()   
        
    # point_id_to_data = {}

    # for line in lines:
    #     if '#' in line:
    #         continue
    #     line = line.replace('\n', '')
    #     POINT3D_ID, X, Y, Z, R, G, B, ERROR, *TRACK = line.split()
    #     point_id_to_data[POINT3D_ID] = [X, Y, Z, R, G, B]


    
            
            
    img_filename_to_num_shared_points = {}
    img_filename_to_close_cameras = {}


    for img1_filename in tqdm(img_filename_to_metadata):
        img_filename_to_close_cameras[img1_filename] = []
        min_num_shared_points_for_img = float('inf')
        
        _, qw1, qx1, qy1, qz1, tx1, ty1, tz1, _ = img_filename_to_metadata[img1_filename]
        
        img1_points_2d = img_filename_to_2d_points[img1_filename]


        for img2_filename in img_filename_to_metadata:
            if img1_filename == img2_filename:
                continue

            img2_points_2d = img_filename_to_2d_points[img2_filename]

        
            matched_observed_point_ids = get_matching_point_ids(img1_points_2d, img2_points_2d)
            
            _, qw2, qx2, qy2, qz2, tx2, ty2, tz2, _ = img_filename_to_metadata[img2_filename]
            
            dist_between_cameras = (((tx1 - tx2) ** 2) + ((ty1 - ty2) ** 2) + ((tz1 - tz2) ** 2)) ** 0.5
            
            if dist_between_cameras > 1:

                continue
            else:
                img_filename_to_close_cameras[img1_filename].append(img2_filename)
                
            min_num_shared_points_for_img = min(min_num_shared_points_for_img, len(matched_observed_point_ids))
            
        img_filename_to_num_shared_points[img1_filename] = min_num_shared_points_for_img
        

    # sort
    img_filename_to_close_cameras = {k: v for k, v in sorted(img_filename_to_close_cameras.items(), key=lambda item: -len(item[1]))}

    
    num_files_processed = 0
    for index, img1_filename in tqdm(enumerate(img_filename_to_close_cameras)):
        if num_files_processed >= args.top_k:
            break
        image_stack_no_alignment = []
        image_stack_homography_alignment = []
        image_stack_stereo_alignment = []
        
        min_num_shared_points_for_img = img_filename_to_num_shared_points[img1_filename]
        matched_filenames = img_filename_to_close_cameras[img1_filename]
        
        
        matched_filenames_filtered = []
        
        # if we don't have the date for the reference image, continue
        if img1_filename.replace('.jpg', '') not in image_id_to_timetaken:
            continue
        matched_filenames_filtered.append(img1_filename)
        
        for filename in matched_filenames:
            if filename.replace('.jpg', '') in image_id_to_timetaken:
                matched_filenames_filtered.append(filename)
        
        print('index', index, 'num matched files:', len(matched_filenames_filtered))
        try:
            matched_filenames_filtered = sorted(matched_filenames_filtered, key=lambda x: image_id_to_timetaken[x.replace('.jpg', '')])
        except KeyError:
            continue

            
        img1_save_path = os.path.join(data_dir, img1_filename)
        img1 = cv.imread(img1_save_path)
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        
        image_stack_no_alignment.append(img1)
        image_stack_homography_alignment.append(img1)
        
        
        for row, img2_filename in enumerate(matched_filenames_filtered):
        
            img2_save_path = os.path.join(data_dir, img2_filename)
            img2 = cv.imread(img2_save_path)
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
            
    #         image_stack_no_alignment.append(img2)
            image_stack_no_alignment.append(cv.resize(img2, dsize=(img1.shape[1], img1.shape[0]), interpolation=cv.INTER_CUBIC))

            img1_points_2d = img_filename_to_2d_points[img1_filename]
            img2_points_2d = img_filename_to_2d_points[img2_filename]
            matched_observed_point_ids = get_matching_point_ids(img1_points_2d, img2_points_2d)
            
            if len(matched_observed_point_ids) <= 4:
                continue
            
            img1_matched_points_mask = np.isin(img1_points_2d[:, 2], matched_observed_point_ids)
            img2_matched_points_mask = np.isin(img2_points_2d[:, 2], matched_observed_point_ids)

            pts_src = img2_points_2d[img2_matched_points_mask] #[:, :2]
            pts_dst = img1_points_2d[img1_matched_points_mask] #[:, :2]

            # sort by 3d point id
            pts_src = pts_src[pts_src[:, 2].argsort()] 
            pts_dst = pts_dst[pts_dst[:, 2].argsort()] 
            
            assert len(pts_src) == len(pts_dst)

            # run ransac
            inlier_cordinates_for_maximum = run_ransac(pts_src, pts_dst, int(args.num_iterations))
            
            if len(inlier_cordinates_for_maximum[0]) < 10:
                continue
            
            inlier_pts_src = pts_src[inlier_cordinates_for_maximum[0]] 
            inlier_pts_dst = pts_dst[inlier_cordinates_for_maximum[0]] 

            
            H, status = cv.findHomography(inlier_pts_src[:, :2], inlier_pts_dst[:, :2], 0)
            im_dst = cv.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
            image_stack_homography_alignment.append(im_dst)

            
        image_stack_no_alignment = np.array(image_stack_no_alignment)
        image_stack_homography_alignment = np.array(image_stack_homography_alignment)
        image_stack_stereo_alignment = np.array(image_stack_stereo_alignment)


        fig, axs = plt.subplots(1, 3, squeeze=False)
        fig.set_figheight(8)
        fig.set_figwidth(20)

        m0 = np.median(image_stack_no_alignment, axis=0).astype(int)
        m1 = np.median(image_stack_homography_alignment, axis=0).astype(int)
        m2 = np.median(image_stack_stereo_alignment, axis=0).astype(int)

        axs[0][0].imshow(img1)
        axs[0][1].imshow(m0)
        axs[0][2].imshow(m1)


        plot_dir = os.path.join('plots', args.dataset, str(args.cluster_id))
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'median_{index}.png'))
        plt.close()


        numpy_dir = os.path.join('numpy_arrays', args.dataset, str(args.cluster_id))
        os.makedirs(numpy_dir, exist_ok=True)
        # save numpy array
        np.save(os.path.join(numpy_dir, f'image_stack_homography_alignment_{index}.npy'), image_stack_homography_alignment)

        num_files_processed += 1




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Align Images", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)

    # TODO: add as arguments
    # DATASET = 'south-building'
    # DATASET = 'rome_colosseum'
    # DATASET = 'trevi_fountain'
    # DATASET = 'trevi_fountain_sample'
    # DATASET = 'stone_henge'
    # args.dataset = 'rialto_bridge'
    # DATASET = 'briksdalsbreen'


    # args.cluster_id = 1