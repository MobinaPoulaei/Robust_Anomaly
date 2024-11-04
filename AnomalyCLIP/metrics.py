from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
import numpy as np
from skimage import measure
import torch
from sklearn.metrics import roc_curve, roc_auc_score


def compute_image_label_from_mask(mask):
    return torch.any(mask).item()


def calculate_max_f1(gt, scores):
    precision, recall, thresholds = precision_recall_curve(gt, scores)
    a = 2 * precision * recall
    b = precision + recall
    f1s = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    index = np.argmax(f1s)
    max_f1 = f1s[index]
    threshold = thresholds[index]
    return max_f1, threshold


def metric_cal(pred_masks, gt_masks, cal_pro=False):
    # Flatten the prediction masks and ground truth masks for pixel-level analysis
    if len(pred_masks.shape) == 4:
        pred_masks = pred_masks.squeeze(1)
    if len(gt_masks.shape) == 4:
        gt_masks = gt_masks.squeeze(1)
    pred_scores_flat = pred_masks.flatten()
    gt_masks_flat = gt_masks.flatten()

    fpr, tpr, _ = roc_curve(gt_masks_flat, pred_scores_flat)
    per_pixel_rocauc = roc_auc_score(gt_masks_flat, pred_scores_flat)

    pxl_f1, pxl_threshold = calculate_max_f1(gt_masks_flat, pred_scores_flat)

    '''
    img_scores = pred_masks.reshape(pred_masks.shape[0], -1).max(axis=1)
    img_labels = np.any(gt_masks.reshape(gt_masks.shape[0], -1), axis=1).astype(int)
    #img_roc_auc = roc_auc_score(img_labels, img_scores)

    # Calculate image-level F1 score and threshold
    img_f1, img_threshold = calculate_max_f1(img_labels, img_scores)
    '''
    pred_binary = (pred_scores_flat >= pxl_threshold).astype(int)  # Binarize predictions based on threshold
    tn, fp, fn, tp = confusion_matrix(gt_masks_flat, pred_binary).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return pxl_f1, sensitivity, specificity


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def cal_confusion_matrix(masks, amaps, max_step=10, expect_fpr=0.3):
    if len(masks.shape) == 4:
        masks = masks.squeeze(1)
    if len(amaps.shape) == 4:
        amaps = amaps.squeeze(1)
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    tps, fps, tns, fns = [], [], [], []

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        tp_pixels = np.logical_and(masks, binary_amaps).sum()
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        tn_pixels = ((binary_amaps == 0) & (masks == 0)).sum()
        fn_pixels = ((binary_amaps == 0) & (masks == 1)).sum()

        tps.append(tp_pixels)
        fps.append(fp_pixels)
        tns.append(tn_pixels)
        fns.append(fn_pixels)

    tps, fps, tns, fns = (np.array(tps), np.array(fps),
                          np.array(tns), np.array(fns))
    tp, fp, tn, fn = np.mean(tps), np.mean(fps), np.mean(tns), np.mean(fns)
    return tp, fp, tn, fn


def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_sp']
    pr = results[obj]['pr_sp']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'image-auroc':
        performance = roc_auc_score(gt, pr)
    elif metric == 'image-ap':
        performance = average_precision_score(gt, pr)
    elif metric == 'image-f1':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        img_scores = pr.reshape(pr.shape[0], -1).max(axis=1)
        img_labels = np.any(gt.reshape(gt.shape[0], -1), axis=1).astype(int)
        performance, img_threshold = calculate_max_f1(img_labels, img_scores)
    return performance
    # table.append(str(np.round(performance * 100, decimals=1)))


def pixel_level_metrics(results, obj, metric):
    gt = results[obj]['imgs_masks']
    pr = results[obj]['anomaly_maps']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'pixel-auroc':
        performance = roc_auc_score(gt.ravel(), pr.ravel())
    elif metric == 'pixel-aupro':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        performance = cal_pro_score(gt, pr)
    elif metric == 'pixel-f1_sensitivity_specificity':
        performance = metric_cal(pr, gt)
    return performance
    