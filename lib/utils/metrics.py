"""
@Fire
https://github.com/fire717
"""
import numpy as np

from config import cfg


def getDist(pre, labels):
    """
    input:
            pre: [batchsize, 14]
            labels: [batchsize, 14]
    return:
            dist: [batchsize, 7]
    """
    pre = pre.reshape([-1, cfg["num_classes"], 2])
    labels = labels.reshape([-1, cfg["num_classes"], 2])
    res = np.power(pre[:, :, 0] - labels[:, :, 0], 2) + np.power(pre[:, :, 1] - labels[:, :, 1], 2)
    return res


def getAccRight(dist, th=5 / cfg['img_size']):
    """
    input:
            dist: [batchsize, 7]

    return:
            dist: [7,]    right count of every point
    """
    res = np.zeros(dist.shape[1], dtype=np.int64)
    for i in range(dist.shape[1]):
        res[i] = sum(dist[:, i] < th)

    return res


def myAcc(output, target):
    """
    return [7,] ndarray
    """
    # print(output.shape) 
    # 64, 7, 40, 40     gaussian
    # (64, 14)                !gaussian
    # b
    # if hm_type == 'gaussian':
    if len(output.shape) == 4:
        output = heatmap2locate(output)
        target = heatmap2locate(target)

    # offset方式还原坐标
    # [h, ls, rs, lb, rb, lr, rr]
    # output[:,6:10] = output[:,6:10]+output[:,2:6]
    # output[:,10:14] = output[:,10:14]+output[:,6:10]

    dist = getDist(output, target)
    cate_acc = getAccRight(dist)
    return cate_acc


def pckh(output, target, threshold=0.5):

    if len(output.shape) == 4:
        output = heatmap2locate(output)
        target = heatmap2locate(target)

    # compute PCK's threshold as percentage of head size in pixels for each pose
    neck_base_joints = (target[:, L_SHOULDER_IND, :] + target[:, R_SHOULDER_IND, :]) / 2
    head_joints = target[:, HEAD_IND, :]
    head_sizes = np.linalg.norm(head_joints - neck_base_joints, axis=2)
    thresholds_head = head_sizes * threshold

    # compute euclidean distances between joints
    distances = np.linalg.norm(output - target, axis=2)

    # compute correct keypoints
    correct_keypoints = (distances <= thresholds_head).astype(int)

    # compute pck
    pck = np.sum(correct_keypoints, axis=0) / correct_keypoints.shape[0]

    return pck
