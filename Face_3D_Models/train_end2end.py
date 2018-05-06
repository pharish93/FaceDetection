import mxnet as mx
import argparse
import pprint

import numpy as np
from face_3d.logger import logger
from face_3d.config import config, default, generate_config
from face_3d.utils.load_data import load_gt_roidb
from face_3d.symbol import *
from face_3d.core.loader import FaceIterator
from face_3d.core.module import Solver
from face_3d.utils.load_param import load_param
from face_3d.core.metric import *


#ctx = mx.gpu(1)
#ctx = mx.cpu()
bgfg = False


def train_net(args, ctx):

    image_set = [args.image_set]


    vgg16_rpn = symbol_vgg16.get_vgg_train()


    ##
    roidbs,face_points_db, classes , mean_face = load_gt_roidb(args.dataset, image_set, args.root_path, args.dataset_path,
                            flip=not args.no_flip)

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

    train_dataiter = FaceIterator(roi_db=roidbs,face_points_db = face_points_db, classes=classes, mean_face= mean_face,
                                  rpn_symbol=vgg16_rpn, interested_labels=interested_labels, rgb_mean=(123.68, 116.779, 103.939),
                                  iterate_range= (0, 17000))

    # eval_dataiter = FaceIterator(roi_db=roidbs, face_points_db=face_points_db, classes=classes, mean_face=mean_face,
    #                              rpn_symbol=vgg16_rpn, interested_labels=interested_labels)

    load_from_vgg = False
    load_proposal_model = True

    if load_from_vgg:
        pretrain_prefix = "model/vgg16"
        pretrain_epoch = 0000
        _, rpn_args, rpn_auxs = mx.model.load_checkpoint(pretrain_prefix, pretrain_epoch)
        rpn_args, rpn_auxs = load_param(ctx[0], vgg16_rpn, rpn_args, rpn_auxs,load_after_vgg=1,load_after_proposal=0)
    elif load_proposal_model:
        rpn_prefix = "model_intermediate/face3d"
        epoch = 20501
        _, rpn_args, rpn_auxs = mx.model.load_checkpoint(rpn_prefix, epoch)
        rpn_args, rpn_auxs = load_param(ctx[0], vgg16_rpn, rpn_args, rpn_auxs,load_after_vgg=0,load_after_proposal=1)


    rpn_model = Solver(
            ctx=ctx[0],
            symbol=vgg16_rpn,
            arg_params=rpn_args,
            aux_params=rpn_auxs,
            optimizer='sgd',
            interested_labels=interested_labels,
            begin_epoch=0,
            num_epoch=50,
            learning_rate=0.001,
            momentum=0.9,
            wd=0.0001,
            bgfg=bgfg,
    )
    rpn_model.fit(
            train_data=train_dataiter,
            eval_data=None,
            regression_metric=smoothl1_metric,
            softmax_metric=softmax_metric,
            epoch_end_callback=mx.callback.do_checkpoint("model_intermediate/face3d")
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train Face 3D Models network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)

    #Harish Modified
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)

    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)

    #Harish Modified for AFLW
    parser.add_argument('--image_set', help='image_set name', default='AFLW', type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    parser.add_argument('--no_flip', help='disable flip images', action='store_true')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    logger.info('Called with arguments: %s' % args)

    #ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    ctx = [mx.cpu()]
    # ctx = [mx.gpu(0)]
    train_net(args, ctx)


if __name__ == "__main__":
    main()
