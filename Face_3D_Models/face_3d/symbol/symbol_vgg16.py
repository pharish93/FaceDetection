import mxnet as mx
from face_3d.config import config

"""
vgg network without final layers
"""


def get_vgg_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")

    return relu5_3


def get_confidence_map(relu_feature, cls_label):
    proposal_cls_score = mx.symbol.Convolution(
        data=relu_feature, kernel=(3, 3), pad=(1, 1), num_filter=11, name="proposal_cls_score"
    )

    proposal_cls_loss = mx.symbol.SoftmaxOutput(*[proposal_cls_score, cls_label], grad_scale=1,
                                                multi_output=True, use_ignore=True, ignore_label=255,
                                                name="proposal_cls_loss")

    return proposal_cls_loss


def get_keypoints_projection(relu_feature, mean_face, proj_weight, proj_label):
    param3d_pred = mx.symbol.Convolution(
        data=relu_feature, kernel=(3, 3), pad=(1, 1), num_filter=8, name="param3d_pred"
    )

    face3dproj = mx.symbol.Face3DProj(
        *[param3d_pred, mean_face], num_keypoints=10, spatial_scale=0.5, name="face3d_proj"
    )

    proj_regression_loss = mx.symbol.SmoothL1Loss(
        *[face3dproj, proj_weight, proj_label], name="proj_regression_loss"
    )

    face3dproj_block = mx.symbol.BlockGrad(data=face3dproj, name="face3dproj_block")
    # box_predict = mx.symbol.BoxPredict(*[face3dproj_block], name="box_predict")

    # return box_predict, proj_regression_loss
    return proj_regression_loss

def do_roi_pooling(relu_feature, box_predict, ground_truth):
    roi_warping = mx.symbol.ROIWarping(*[relu_feature, box_predict, ground_truth], warped_shape=(28, 28),
                                       spatial_scale=0.5, name="roi_warping")
    roi_warping_pool = mx.symbol.Pooling(
        data=roi_warping, pool_type="max", kernel=(4, 4), stride=(4, 4), name="roi_warping_pool"
    )
    roi_warping_flatten = mx.symbol.Flatten(data=roi_warping_pool)

    return roi_warping_flatten


## Harish : Added to experiment

def compute_fc_layers(roi_warping_flatten):
    roi_warping_fc1 = mx.symbol.FullyConnected(data=roi_warping_flatten, num_hidden=1024, name="roi_warping_fc1")
    roi_warping_bn1 = mx.symbol.BatchNorm(data=roi_warping_fc1, name="roi_warping_bn1")
    roi_warping_relu1 = mx.symbol.Activation(data=roi_warping_bn1, act_type="relu", name="roi_warping_relu1")

    roi_warping_fc2 = mx.symbol.FullyConnected(data=roi_warping_relu1, num_hidden=1024, name="roi_warping_fc2")
    roi_warping_bn2 = mx.symbol.BatchNorm(data=roi_warping_fc2, name="roi_warping_bn2")
    roi_warping_relu2 = mx.symbol.Activation(data=roi_warping_bn2, act_type="relu", name="roi_warping_relu2")

    return roi_warping_relu2


def get_vgg_train():
    data = mx.symbol.Variable(name="data")
    mean_face = mx.symbol.Variable(name="mean_face")

    # confidence map
    cls_label = mx.symbol.Variable(name="cls_label")

    # projection regression
    proj_weight = mx.symbol.Variable(name="proj_weight")
    proj_label = mx.symbol.Variable(name="proj_label")

    # roi pooling
    ground_truth = mx.symbol.Variable(name="ground_truth")

    # bbox regression
    ellipse_label = mx.symbol.Variable(name="ellipse_label")
    bbox_weight = mx.symbol.Variable(name="bbox_weight")

    # shared convolutional layers -- VGG 16
    relu5_3 = get_vgg_conv(data)

    relu5_3_block = mx.symbol.BlockGrad(data=relu5_3, name="relu5_3_block")

    # Face 3d Network
    # Deconvolution
    upsample_feature = mx.symbol.Deconvolution(
        data=relu5_3_block, kernel=(16, 16), stride=(8, 8), pad=(4, 4), num_filter=512, num_group=512,
        no_bias=True, name="upsample_proposal", workspace=4096
    )

    conv_feature = mx.symbol.Convolution(
        data=upsample_feature, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv_proposal")
    # conv_feature_bn1 = mx.symbol.BatchNorm(data=conv_feature, name="conv_feature_bn1")
    relu_feature = mx.symbol.Activation(
        data=conv_feature, act_type="relu", name="relu_proposal")

    # relu_feature_block = mx.symbol.BlockGrad(data=relu_feature, name="relu_feature_block")


    # confidence map
    proposal_cls_loss = get_confidence_map(relu_feature, cls_label)

    # face keypoints projection
    proj_regression_loss = get_keypoints_projection(relu_feature, mean_face, proj_weight, proj_label)
    # ell_label = mx.symbol.GenEllLabel(*[box_predict, ellipse_label, ground_truth], spatial_scale=0.5, name="ell_label")
    #
    # # roi warping
    #
    # roi_warping_flatten = do_roi_pooling(relu_feature, box_predict, ground_truth)
    #
    # # FC layers
    #
    # roi_warping_relu2 = compute_fc_layers(roi_warping_flatten)
    #
    # # bbox prediction
    # offset_predict = mx.symbol.FullyConnected(data=roi_warping_relu2, num_hidden=5, name="offset_predict")
    #
    # ellipse_predict_loss = mx.symbol.SmoothL1Loss(
    #     *[offset_predict, bbox_weight, ell_label], name="ellipse_predict_loss"
    # )

    # loss_all = mx.symbol.Group([proposal_cls_loss, proj_regression_loss, ellipse_predict_loss, ell_label, box_predict])
    # loss_all = mx.symbol.Group([proposal_cls_loss, proj_regression_loss, box_predict])
    loss_all = mx.symbol.Group([proposal_cls_loss, proj_regression_loss])

    # loss_all = mx.symbol.Group([proj_regression_loss])

    return loss_all
