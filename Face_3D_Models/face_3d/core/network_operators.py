import numpy as np

INF = 0x3f3f3f3f
iou_self_threshold = 0.8


def calc_ground_truth(height, width, num_cls, tensor_keypoint, tensor_softmax, argmax_label,spatial_scale):
    tensor_keypoint = tensor_keypoint.reshape((height, width, num_cls - 1, 2))
    ground_truth = np.zeros((height * width, 2))
    faceness = np.zeros((height * width))
    num_ground_truth = 0

    for i in xrange(height):
        for j in xrange(width):
            if argmax_label[i, j] != 0:
                ground_truth[num_ground_truth] = np.array([i * 2, j * 2])
                cnt_inside_img = 0
                for k in xrange(num_cls - 1):
                    predict_x = int(tensor_keypoint[i, j, k, 0] * spatial_scale)
                    predict_y = int(tensor_keypoint[i, j, k, 1] * spatial_scale)
                    if predict_x >= 0 and predict_x < height and predict_y >= 0 and predict_y < width and tensor_softmax[k + 1, predict_x, predict_y] != 0.0 :
                        faceness[num_ground_truth] += np.log(tensor_softmax[k + 1, predict_x, predict_y])
                        cnt_inside_img += 1

                if cnt_inside_img < 3:
                    faceness[num_ground_truth] = -INF
                else:
                    faceness[num_ground_truth] = faceness[num_ground_truth] / cnt_inside_img * (num_cls - 1)

                num_ground_truth += 1

    return ground_truth[:num_ground_truth], faceness[:num_ground_truth]


def calc_iou_rectangle(rec0, rec1):
    tempt = np.array([rec0, rec1])
    l, u = tempt.max(axis=0)[0:2]
    r, d = tempt.min(axis=0)[2:4]
    if l >= r or u >= d:
        return 0

    insec = (r - l) * (d - u)
    area = (rec0[2] - rec0[0]) * (rec0[3] - rec0[1]) + \
           (rec1[2] - rec1[0]) * (rec1[3] - rec1[1]) - insec
    if area == 0:
        return 0
    return float(insec) / float(area)


def non_maximum_suppression_rpn(ground_truth, faceness, box_predict, spatial_scale):
    num_gt = ground_truth.shape[0]
    ret_gt = np.zeros((num_gt, 2), dtype=np.float32)
    ret_faceness = np.zeros(num_gt, dtype=np.float32)
    num_gt_ret = 0

    rec_ret = np.zeros((num_gt, 4), dtype=np.float32)

    for i in xrange(num_gt):
        if faceness[i] < -100:
            break

        flag = True
        rec0 = np.array(box_predict[0,
                        int(ground_truth[i, 0] * spatial_scale),
                        int(ground_truth[i, 1] * spatial_scale), :])
        rec0[2] += rec0[0]
        rec0[3] += rec0[1]

        for j in xrange(num_gt_ret):
            if calc_iou_rectangle(rec0, rec_ret[j]) > iou_self_threshold:
                flag = False
                break

        if flag:
            ret_gt[num_gt_ret] = ground_truth[i]
            ret_faceness[num_gt_ret] = faceness[i]
            rec_ret[num_gt_ret] = np.array(box_predict[0,
                                           int(ret_gt[num_gt_ret, 0] * spatial_scale),
                                           int(ret_gt[num_gt_ret, 1] * spatial_scale), :])
            rec_ret[num_gt_ret, 2] += rec_ret[num_gt_ret, 0]
            rec_ret[num_gt_ret, 3] += rec_ret[num_gt_ret, 1]
            num_gt_ret += 1

    return ret_gt[:num_gt_ret], ret_faceness[:num_gt_ret]


def calc_ellipse(ell_output, predict_bbox_output, ground_truth, spatial_scale):
    ellipses = np.zeros((ell_output.shape[0], 5), dtype=np.float)
    for i in xrange(ell_output.shape[0]):
        predict_bbox = predict_bbox_output[0,
                       int(ground_truth[i, 0] * spatial_scale),
                       int(ground_truth[i, 1] * spatial_scale), :]
        ellipses[i] = np.array([ell_output[i, 0] * predict_bbox[2] * 2,
                                ell_output[i, 1] * predict_bbox[3] * 2,
                                ell_output[i, 4],
                                ell_output[i, 2] * predict_bbox[2] + predict_bbox[0],
                                ell_output[i, 3] * predict_bbox[3] + predict_bbox[1]])

    return ellipses

