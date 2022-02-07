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


def pckh(output, target, head_sizes, threshold=0.5):

    if len(output.shape) == 4:
        output = heatmap2locate(output)
        target = heatmap2locate(target)

    # reshape output and target to [batch_size, joints_num, 2]
    output = output.reshape([len(output), -1, 2])
    target = target.reshape([len(target), -1, 2])

    # compute PCK's threshold as percentage of head size in pixels for each pose
    thresholds_head = head_sizes * threshold
    thresholds_head = thresholds_head.reshape([-1, 1]).tile((1, target.shape[1]))

    # compute euclidean distances between joints
    distances = np.linalg.norm(output - target, axis=2)

    # compute correct keypoints
    correct_keypoints = (distances <= np.array(thresholds_head)).astype(int)

    # remove not annotated keypoints from pck computation
    correct_keypoints = correct_keypoints * (target[:, :, 0] != -1).astype(int)
    annotated_keypoints_num = np.sum((target[:, :, 0] != -1).astype(int), axis=0)

    # compute pck
    # pckh_joints = np.sum(correct_keypoints, axis=0) / correct_keypoints.shape[0]
    pck_joints = np.sum(correct_keypoints, axis=0) / annotated_keypoints_num
    pck_avg = np.mean(pck_joints)

    return pck_joints, pck_avg
