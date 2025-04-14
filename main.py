import numpy as np
from scipy.stats import binom, norm
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from scipy import ndimage
from pathlib import Path
import cv2
import os
from imutils.object_detection import non_max_suppression
import argparse

ROOT = os.path.dirname(os.path.realpath(__file__))

def load_image(img_path):
    """
    Loads grayscale image from a given path
    """
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    return img

def binarize_image(img, nb_classes=4):
    """
    Binarizes a grayscale image using the smallest threshold derived from  
    the multi-otsu method with nb_classes classes
    """
    thresh = threshold_multiotsu(img, classes = nb_classes)[0]
    return img < thresh

def extract_components(img, blob_thresh=10):
    """
    Extracts labeled connected components from the binarized image, removing  
    components with area smaller than blob_thresh
    """
    binary= binarize_image(img)
    # Label connected components
    labeled_objects, _ = ndimage.label(binary)
    
    # Remove components whose area is smaller than blob_thresh 
    counts = np.bincount(labeled_objects.ravel())
    valid_labels = np.where(counts > blob_thresh)[0]
    valid_labels = valid_labels[1:]
    # Create a mask with only the valid labels
    valid_objects = np.isin(labeled_objects, valid_labels) * labeled_objects

    return valid_objects, valid_labels.tolist()


def compute_std_per_label(img, labeled_objects, labels):
    """
    Computes standard deviation for each labeled region 
    """
    stds = ndimage.labeled_comprehension(img, labeled_objects, labels, np.std, float, 0)

    # Create an image with standard deviations assigned to each region
    stds_img = np.zeros_like(img, dtype=np.float32)
    for i, std in zip(labels, stds):
        stds_img[labeled_objects == i] = std
    
    stds_img -= stds_img.min()
    stds_img /= stds_img.max()

    return stds, stds_img

def compute_outliers_mask(label_objects, labels, stds, alpha=0.1):
    """
    Computes a statistical mask using standard deviation filtering.
    """
    mean_stds = np.mean(stds)
    sigma_stds = np.std(stds)
    z = norm.isf(alpha)
    thresh = mean_stds - z * sigma_stds

    # Create mask where standard deviation is below threshold
    mask = np.isin(label_objects, [i for i, std in zip(labels, stds) if std < thresh]).astype(np.uint8) * 255

    return mask

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

def compute_NFA(mask, chars, text_mask, threshold, alpha):
    label_words, nb_words = ndimage.label(text_mask)
    NFADets = np.zeros_like(text_mask, dtype=np.uint8)
    for label in range(1, nb_words+1):
        word_mask = (label_words == label)
        _, tot = ndimage.label(word_mask & chars)
        _, dets = ndimage.label(word_mask & mask)
        NFA = nb_words * (1 - binom.cdf(dets-1, tot, alpha))
        if NFA < threshold:
            NFADets[(word_mask & chars)>0] = 1
    return NFADets

def detect(img_path, alpha, threshold):
    img = load_image(img_path)
    labeled_objects, labels = extract_components(img)
    chars = (labeled_objects > 0)
    stds, stds_img = compute_std_per_label(img, labeled_objects, labels)
    outliers = compute_outliers_mask(labeled_objects, labels, stds, alpha)
    east_model_path = os.path.join( ROOT, 'frozen_east_text_detection.pb')
    text_mask = detect_text(img, east_model_path)
    nfa = compute_NFA(outliers, chars, text_mask, threshold, alpha)
    cv2.imwrite('characters.png', chars * 255)
    cv2.imwrite('outliers.png', outliers)
    cv2.imwrite('words.png', text_mask)
    cv2.imwrite('nfa.png', nfa * 255)
    cv2.imwrite('stds.png', stds_img * 255)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    #parser.add_argument("mask")
    parser.add_argument("-a")
    parser.add_argument("-t")
    parser.parse_args()
    args = parser.parse_args()
    img_path = args.image
    #mask_path = args.mask
    alpha = float(args.a)
    trheshold = float(args.t)
    detect(img_path, alpha, trheshold)
    #detect(img_path, mask_path, alpha, trheshold)