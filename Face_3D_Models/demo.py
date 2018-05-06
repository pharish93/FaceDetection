import numpy as np
import cv2
import mxnet as mx
import logging

from face_3d.utils.load_param import load_param
from face_3d.symbol import *
from face_3d.utils.visualizations import make_heat_map,vis_proposal_after_nms


# 0 LeftEyeLeftCorner
# 1 RightEyeRightCorner
# 2 LeftEar
# 3 NoseLeft
# 4 NoseRight
# 5 RightEar
# 6 MouthLeftCorner
# 7 MouthRightCorner
# 8 ChinCenter
# 9 center_between_eyes

INF = 0x3f3f3f3f
spatial_scale = 0.5
num_classes = 11

logger = logging.getLogger()
logger.setLevel(logging.INFO)
rgb_mean = np.array([123.68, 116.779, 103.939])
num_class = 10


def calc_ground_truth(height, width, num_cls, tensor_keypoint, tensor_softmax, argmax_label):
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
                    if predict_x >= 0 and predict_x < height and predict_y >= 0 and predict_y < width:
                        faceness[num_ground_truth] += np.log(tensor_softmax[k + 1, predict_x, predict_y])
                        cnt_inside_img += 1

                if cnt_inside_img < 3:
                    faceness[num_ground_truth] = -INF
                else:
                    faceness[num_ground_truth] = faceness[num_ground_truth] / cnt_inside_img * (num_cls - 1)

                num_ground_truth += 1

    return ground_truth[:num_ground_truth], faceness[:num_ground_truth]


def load_mean_face(interested_classes=None):

    f_class = open("demo_files/class.txt", 'r')
    count = 1
    class_names = {}
    class_names[0] = "background"
    for line in f_class:
        class_names[line.strip('\n')] = count
        class_names[count] = line.strip('\n')
        count += 1

    f_class.close()

    file_mface = "demo_files/model3d.txt"
    mean_face = np.zeros([num_class, 3], np.float32)
    f_mface = open(file_mface, 'r')
    for line in f_mface:
        line = line.strip('\n').split(' ')
        xyz = np.array([float(line[1]), float(line[2]), float(line[3])])
        mean_face[class_names[line[0]] - 1] = xyz
    f_mface.close()

    return mean_face


def demo(ctx, demo_file_location, image_file_name):

    interested_labels = ['background',
                         'LeftEyeLeftCorner',
                         'RightEyeRightCorner',
                         'LeftEar',
                         'NoseLeft',
                         'NoseRight',
                         'RightEar',
                         'MouthLeftCorner',
                         'MouthRightCorner',
                         'ChinCenter',
                         'center_between_eyes']

    mean_face = load_mean_face(interested_labels)

    load_proposal_model = True

    vgg16_rpn = symbol_vgg16.get_vgg_train()
    if load_proposal_model:
        rpn_prefix = "model_intermediate/face3d"
        epoch = 20501
        _, rpn_args, rpn_auxs = mx.model.load_checkpoint(rpn_prefix, epoch)
        rpn_args, rpn_auxs = load_param(ctx[0], vgg16_rpn, rpn_args, rpn_auxs, load_after_vgg=0, load_after_proposal=1)

    ctx= ctx[0]
    rpn_args["mean_face"] = mx.nd.array(mean_face, ctx)

    imgFilename = demo_file_location + image_file_name

    img_bgr = cv2.imread(imgFilename)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    width, height = img.shape[1],img.shape[0]
    short_len = min(width, height)
    resize_factor = 600.0 / float(short_len)
    long_len = max(width * resize_factor, height * resize_factor)
    if long_len > 1000:
        resize_factor = 1000.0 / float(max(width, height))

    width = int(resize_factor * width)
    height = int(resize_factor * height)
    out_height, out_width = vgg16_rpn.infer_shape(data=(1, 3, height, width), mean_face=(10, 3),
                                                        proj_label=(height, width, 10, 2))[1][0][2:4]
    print height, width, out_height, out_width

    cls_label = mx.nd.empty((1, out_height * out_width), ctx)
    proj_label = mx.nd.empty((1, out_height, out_width, 10, 2), ctx)
    proj_weight = mx.nd.empty((1, out_height, out_width, 10, 2), ctx)
    rpn_args["cls_label"] = cls_label
    rpn_args["proj_label"] = proj_label
    rpn_args["proj_weight"] = proj_weight

    img = cv2.resize(img, (width, height))
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = np.array(img, dtype=np.float32)
    img = img - rgb_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # (c,h,w)
    img = np.expand_dims(img, axis=0)  # (1,c,h,w) 1 is batch number

    rpn_args["data"] = mx.nd.array(img, ctx)

    executor = vgg16_rpn.bind(ctx, rpn_args, args_grad=None, grad_req="null", aux_states=rpn_auxs)
    executor.forward(is_train=True)

    softmax_output = mx.nd.zeros(executor.outputs[0].shape)
    regression_output = mx.nd.zeros(executor.outputs[1].shape)
    box_predict_output = mx.nd.zeros(executor.outputs[2].shape)
    executor.outputs[0].copyto(softmax_output)
    executor.outputs[1].copyto(regression_output)
    executor.outputs[2].copyto(box_predict_output)
    softmax_output = np.squeeze(softmax_output.asnumpy())
    regression_output = np.squeeze(regression_output.asnumpy())
    argmax_label = np.uint8(softmax_output.argmax(axis=0))
    out_img = np.uint8(softmax_output.argmax(axis=0))
    box_predict_output = box_predict_output.asnumpy()

    heat_map_name = "./demo_files/heat_map"+image_file_name
    make_heat_map(argmax_label, 11 ,0,op_filename= heat_map_name)

    ground_truth, tmp_faceness = calc_ground_truth(out_height, out_width, num_classes, regression_output,
                                                   softmax_output, argmax_label)

    print ground_truth.shape

    proposals_name = "./demo_files/proposals_"+image_file_name
    vis_proposal_after_nms(img, ground_truth, box_predict_output, 0, proposals_name)


def main():
    print " Starting the Demo"

    demo_file_location = "./demo_files/"
    image_file_name =  "demo_input.jpg"
    ctx = [mx.cpu()]
    # ctx = [mx.gpu(0)]
    demo(ctx, demo_file_location, image_file_name)


if __name__ == "__main__":
    main()
