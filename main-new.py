#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 00:06:17 2024

@author: marina
"""

#TODO: Merge the computation of the NFA with detect function

import numpy as np
from scipy.stats import binom, norm
from skimage import io
import os
import glob
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from scipy import ndimage
# import mpmath
from pathlib import Path
import concurrent.futures
import argparse
from tqdm import tqdm
import cv2
from imutils.object_detection import non_max_suppression

def detect_text(image, east_model_path, min_confidence=0.1):
    """
    Text detection using the EAST model.
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    orig_h, orig_w = image.shape[:2]
    new_w, new_h = (int(np.ceil(orig_w / 32) * 32), int(np.ceil(orig_h / 32) * 32))
    rW, rH = orig_w / float(new_w), orig_h / float(new_h)

    resized = cv2.resize(image, (new_w, new_h))

    blob = cv2.dnn.blobFromImage(resized, 1.0, (new_w, new_h),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net = cv2.dnn.readNet(east_model_path)

    layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    net.setInput(blob)
    scores, geometry = net.forward(layer_names)

    num_rows, num_cols = scores.shape[2:4]
    rects, confidences = [], []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0, x1, x2, x3 = (geometry[0, i, y] for i in range(4))
        angles = geometry[0, 4, y]

        for x in range(num_cols):
            score = scores_data[x]
            if score < min_confidence:
                continue

            angle = angles[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h, w = x0[x] + x2[x], x1[x] + x3[x]
            offsetX, offsetY = x * 4.0, y * 4.0

            endX = int(offsetX + cos * x1[x] + sin * x2[x])
            endY = int(offsetY - sin * x1[x] + cos * x2[x])
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(score)

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    text_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    for (startX, startY, endX, endY) in boxes:
        startX = max(0, int(startX * rW))
        startY = max(0, int(startY * rH))
        endX = min(orig_w, int(endX * rW))
        endY = min(orig_h, int(endY * rH))
        text_mask[startY:endY, startX:endX] = 255

    return text_mask

def get_characters(img, blob_thresh=30):
    binary = np.ones_like(img[:,:,0])
    for c in range(3):
        #binary_ch = np.ones_like(img[:,:,ch])
        thresh = threshold_multiotsu(img[:,:,c], classes=4)[0]
        #print(f'{ch} {thresh}')
        binary_ch = img[:,:,c] < thresh 
        binary *= binary_ch
    label_objects, nb_labels = ndimage.label(binary)
    new_labels = []
    new_label_objects = label_objects.copy()
    print(f'Number of labels: {nb_labels}')
    for i in tqdm(range(1,nb_labels+1)):
        if np.sum(label_objects==i) > blob_thresh:
            new_labels.append(i)
        else:
            new_label_objects[np.where(label_objects==i)]=0
    return new_label_objects, new_labels

def compute_std(args):
    img_ch, label_objects, i = args
    std = np.std(img_ch[np.where(label_objects==i)])
    return std, i

def detect_three_channels(img_path, res_path):
    #read inputs
    sub_folder = os.path.splitext(os.path.basename(img_path))[-2]
    if not os.path.exists(res_path + '/' + sub_folder):
        os.makedirs(res_path + '/' + sub_folder)
    img = io.imread(img_path)
    mask= np.ones(img[:,:,0].shape)
    mask_bonferroni= np.ones(img[:,:,0].shape)
    label_objects, labels = get_characters(img)
    labels_binary = label_objects>0
    io.imsave(f'{res_path + '/' + sub_folder}/characters.png', 255*labels_binary.astype(np.uint8))
    for ch in range(3):
        stds = []
        img_ch = img[:,:,ch]
        stds_img = np.zeros(label_objects.shape)
        for i in labels:
            std = np.std(img_ch[np.where(label_objects==i)])
            stds.append(std)
            stds_img[np.where(label_objects==i)] = std
        io.imsave(f'{res_path + '/' + sub_folder}/stds_{ch}.tiff', stds_img)
        mean_stds = np.mean(stds)
        sigma_stds = np.std(stds)
        mask_ch = np.zeros(label_objects.shape)
        mask_ch_bonferroni = np.zeros(label_objects.shape)
        stds = np.array(stds)
        alpha = 0.1
        z = norm.isf(alpha)
        thresh = mean_stds - z* sigma_stds
        for k,i in enumerate(labels):
            std = stds[k]
            if std < thresh:
                mask_ch[np.where(label_objects==i)] = 255
        mask *= mask_ch
        #z = norm.isf(alpha/len(labels))
        #thresh = mean_stds - z * sigma_stds
        #for k,i in enumerate(labels):
        #    std = stds[k]
        #    if std < thresh:
        #        mask_ch_bonferroni[np.where(label_objects==i)] = 255
        #mask_bonferroni *= mask_ch_bonferroni
        io.imsave(f'{res_path + '/' + sub_folder}/mask_{ch}.png', mask_ch.astype(np.uint8))
        #io.imsave(f'{res_path}/mask_{ch}_bonferroni.png', mask_ch_bonferroni.astype(np.uint8))
    io.imsave(f'{res_path + '/' + sub_folder}/mask.png', mask.astype(np.uint8))
    #io.imsave(f'{res_path}/mask_bonferroni.png', mask_bonferroni.astype(np.uint8))
    #return mask

def compute_NFA(img_path, res_path):
    filename =  Path(img_path).stem
    sub_folder = os.path.splitext(os.path.basename(img_path))[-2]
    Char = io.imread(f'{res_path + '/' + sub_folder}/characters.png')/255
    #text_mask = io.imread('/Users/julietaumpierrez/Desktop/iccv-doc-workshop/NoiseDFD/noiseFD-outputs/tampered/'+ filename +'/words.png')
    east_model_path = '/Users/julietaumpierrez/Desktop/iccv-doc-workshop/NoiseDFD/frozen_east_text_detection.pb'
    img = cv2.imread(img_path)
    text_mask = detect_text(img, east_model_path)
    io.imsave(f'{res_path + '/' + sub_folder}/text-mask.png', text_mask.astype(np.uint8))
    label_words, nb_words = ndimage.label(text_mask)
    detChar = io.imread(f'{res_path + '/' + sub_folder}/mask.png')
    NFADets = np.zeros(text_mask.shape)
    for label in range(1, nb_words+1):
        word_mask = np.zeros(text_mask.shape)
        word_mask[np.where(label_words==label)] = 1
        _, dets = ndimage.label(word_mask*detChar)
        _, tot = ndimage.label(word_mask*Char)
        #print(nb_words)
        NFA = nb_words * (1 - binom.cdf(dets-1, tot, 0.1))
        if NFA < 0.01:
            print(dets, tot)
            NFADets[np.where(word_mask*Char>0)] = 1
    if not os.path.exists(res_path + '/' + sub_folder):
        os.makedirs(res_path + '/' + sub_folder)
    io.imsave(f'{res_path + '/' +sub_folder}/nfa.png', 255*NFADets.astype(np.uint8))


# def main(im_path):
#     img_name = os.path.basename(im_path).strip('.tif')
#     print(f'Processing {img_name}')
#     res_directory = f'results-prueba/{img_name}'
#     Path(res_directory).mkdir(parents=True, exist_ok=True)
#     #detect_three_channels(im_path, res_directory)
#     compute_NFA(im_path, res_directory)
#     #compute_NFA_new(im_path, res_directory)
#     print(f'Finished {img_name}')

# main("/home/mgardella/Research/Doc-forensics/CollabNextForgeryDetection/variance-analysis/s11_27_c.tif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("output_path")
    parser.parse_args()
    args = parser.parse_args()
    path = args.dataset_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #mask_path = args.mask

    for img in Path(path).glob('*.tif'):
        print(f"Processing {img}")
        sub_folder = os.path.splitext(os.path.basename(str(img)))[-2]
        # If mask and characters already exist skip detect three channels
        if os.path.exists(os.path.join(output_path, sub_folder, 'mask.png')) and \
           os.path.exists(os.path.join(output_path, sub_folder, 'characters.png')):
            print(f"Skipping {img} as results already exist.")
            compute_NFA(str(img), output_path)
        else:
            detect_three_channels(str(img), output_path)
            compute_NFA(str(img), output_path)

# if __name__ == '__main__':
#     print('Starting')
#     img_paths = glob.glob('../../doc-forgery-det/datasets/supatlantique/Retouching/*.tif')
#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#         futures = {executor.submit(main, img_path) for img_path in img_paths}
#         for f in glob.glob('../../doc-forgery-det/datasets/supatlantique/Retouching/*.tif'):
#             #print(f)
#             executor.submit(main, f)

# for f in glob.glob('/Users/antoine/Documents/CollabNext/Datasets/SUPATLANTIQUE/Tampered/Retouching/*.tif')[:1]:
#     img_name = os.path.basename(f).strip('.tif')
#     res_directory = f'new_results-compare/{img_name}'
#     Path(res_directory).mkdir(parents=True, exist_ok=True)
#     detect_three_channels(f, res_directory)
#     compute_NFA(f, res_directory)