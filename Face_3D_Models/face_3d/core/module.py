import numpy as np
import mxnet as mx
import logging
from collections import namedtuple
from mxnet import optimizer as opt
from mxnet.optimizer import get_updater
from mxnet import metric
from face_3d.core.metric import *
from face_3d.utils.visualizations import *
from face_3d.core.network_operators import *

BatchEndParam = namedtuple('BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])



class Solver(object):
    def __init__(self, symbol,
                 ctx=None,
                 begin_epoch=0, num_epoch=0,
                 arg_params=None, aux_params=None, bgfg=False,
                 interested_labels=[],
                 optimizer='sgd', **kwargs):

        self.grad_params = None
        self.symbol = symbol
        if ctx is None:
            ctx = mx.cpu()
        self.ctx = ctx
        self.begin_epoch = begin_epoch
        self.num_epoch = num_epoch
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.optimizer = optimizer
        self.kwargs = kwargs.copy()
        self.bgfg = bgfg
        self.interested_labels = interested_labels
        self.num_classes = len(self.interested_labels)

    def network_backprop_setup(self, grad_req, arg_names, arg_shapes, eval_metric):

        if grad_req != 'null':
            self.grad_params = {}
            for name, shape in zip(arg_names, arg_shapes):
                if not (name.endswith('data') or name.endswith("mean_face") or name.endswith('cls_label') or
                            name.endswith('proj_weight') or name.endswith('proj_label')or name.endswith('ground_truth') or
                        name.endswith('ellipse_label') or name.endswith("bbox_weight")):
                    self.grad_params[name] = mx.nd.zeros(shape, self.ctx)

        # setting the required optimizer
        self.optimizer = opt.create(self.optimizer, rescale_grad=1.0, **(self.kwargs))
        self.updater = get_updater(self.optimizer)
        eval_metric = metric.create(eval_metric)

        return eval_metric

    def network_input_setup(self, train_data, data):

        data_name = train_data.data_name
        cls_label_name = train_data.cls_label_name

        proj_label_name = train_data.proj_label_name
        proj_weight_name = train_data.proj_weight_name

        ground_truth_name = train_data.ground_truth_name

        ellipse_label_name = train_data.ellipse_label_name
        bbox_weight_name = train_data.bbox_weight_name

        softmax_shape = data[cls_label_name].shape
        self.arg_params[data_name] = mx.nd.array(data[data_name], self.ctx)
        self.arg_params[cls_label_name] = mx.nd.array(
            data[cls_label_name].reshape((softmax_shape[0], softmax_shape[1] * softmax_shape[2])), self.ctx)

        self.arg_params[proj_label_name] = mx.nd.array(data[proj_label_name], self.ctx)
        self.arg_params[proj_weight_name] = mx.nd.array(data[proj_weight_name], self.ctx)

        self.arg_params[ground_truth_name] = mx.nd.array(data[ground_truth_name], self.ctx)

        self.arg_params[ellipse_label_name] = mx.nd.array(data[ellipse_label_name], self.ctx)
        self.arg_params[bbox_weight_name] = mx.nd.array(data[bbox_weight_name], self.ctx)

        self.arg_params["mean_face"] = mx.nd.array(train_data.mean_face, self.ctx)


    def proj_regression_loss_analysis(self, train_data, data, output_buff, regression_metric=None):

        proj_label_name = train_data.proj_label_name
        proj_weight_name = train_data.proj_weight_name
        proj_label = data[proj_label_name]
        proj_weight = data[proj_weight_name]
        proj_pred = output_buff["proj_regression_loss_output"].asnumpy() \
            .reshape(data[proj_label_name].shape)
        index_nonzero = np.nonzero(data[proj_weight_name])

        proj_regress_tmp = 0
        if regression_metric:
            proj_regress_tmp = regression_metric(proj_label[index_nonzero], proj_pred[index_nonzero],
                                                 proj_weight[index_nonzero])

        return proj_regress_tmp

    def classification_loss_analysis(self, train_data, data, output_buff, softmax_metric=None):

        cls_label_name = train_data.cls_label_name
        softmax_shape = data[cls_label_name].shape

        pred_shape = output_buff["proposal_cls_loss_output"].shape
        index_label = np.nonzero(data[cls_label_name]
                                 .reshape(softmax_shape[0], softmax_shape[1] * softmax_shape[2]) - 255)
        label = mx.nd.array(data[cls_label_name].reshape(softmax_shape[0], softmax_shape[1] * softmax_shape[2])
                            [:, index_label[1]])
        pred = mx.nd.array((output_buff["proposal_cls_loss_output"].asnumpy()
                            .reshape(pred_shape[0], pred_shape[1], pred_shape[2] * pred_shape[3]))
                           [..., index_label[1]])

        tempt = 0

        if softmax_metric:
            tempt = softmax_metric(label, pred, 11)

        return tempt

    def print_batch_logs(self, f, img_pos, train_data, epoch, nbatch, softmax_loss_tmp, proj_regress_tmp,bbox_predict_tmp):

        img_info = train_data.AllImg[img_pos]
        print "%s\twidth: %d height: %d num_face: %d" % \
              (img_info.filename, img_info.width, img_info.height, img_info.num_faces)
        f.write("%s\twidth: %d height: %d num_face: %d\n" % \
                (img_info.filename, img_info.width, img_info.height, img_info.num_faces))

        print"Training-epoch[%d]-batch[%d]: acc:%f\tproj_regress:%f\tbbox_regress:%f\tbbox_angle:%f" % \
              (epoch, nbatch, get_accuracy(softmax_loss_tmp, self.bgfg), proj_regress_tmp,
               bbox_predict_tmp[0], bbox_predict_tmp[1])

        f.write("Training-epoch[%d]-batch[%d]: acc:%f\tproj_regress:%f\tbbox_regress:%f\tbbox_angle:%f\n" %
                (epoch, nbatch, get_accuracy(softmax_loss_tmp, self.bgfg), proj_regress_tmp,
                bbox_predict_tmp[0], bbox_predict_tmp[1]))


        # print"Training-epoch[%d]-batch[%d]: acc:%f\tproj_regress:%f\tbbox_regress:%f\tbbox_angle:%f" % \
        #       (epoch, nbatch, 0, proj_regress_tmp,
        #        bbox_predict_tmp[0], bbox_predict_tmp[1])
        #
        # f.write("Training-epoch[%d]-batch[%d]: acc:%f\tproj_regress:%f\tbbox_regress:%f\tbbox_angle:%f\n" %
        #         (epoch, nbatch, 0, proj_regress_tmp,
        #         bbox_predict_tmp[0], bbox_predict_tmp[1]))

    def visualize_intermediate_outputs(self, train_data, data, output_buff, nbatch, predict_data = None):

        heat_map = output_buff["proposal_cls_loss_output"].asnumpy().argmax(axis=1)[0]
        make_heat_map(heat_map, self.num_classes, nbatch)

        proj_label_name = train_data.proj_label_name
        proj_label = data[proj_label_name]
        proj_pred = output_buff["proj_regression_loss_output"].asnumpy() \
            .reshape(data[proj_label_name].shape)
        vis_proposals(data[train_data.data_name], proj_label, proj_pred, nbatch)
        # box_predict_output = output_buff["box_predict_output"].asnumpy()
        # ground_truth = data[train_data.ground_truth_name]
        # vis_proposal_after_nms(data[train_data.data_name], ground_truth, box_predict_output, nbatch)


        if predict_data:
            softmax_shape = data[train_data.cls_label_name].shape
            regression_output = np.squeeze(output_buff["proj_regression_loss_output"].asnumpy())
            softmax_output = np.squeeze(output_buff["proposal_cls_loss_output"].asnumpy())
            argmax_label = np.uint8(softmax_output.argmax(axis=0))
            spatial_scale = 0.5
            ground_truth, tmp_faceness = calc_ground_truth(softmax_shape[1],softmax_shape[2], 11, regression_output,
                                                           softmax_output, argmax_label,spatial_scale)

            index_lst = np.argsort(-tmp_faceness)
            ground_truth = ground_truth[index_lst, :]
            tmp_faceness = tmp_faceness[index_lst]
            box_predict_output = output_buff["box_predict_output"].asnumpy()
            ground_truth, tmp_faceness = non_maximum_suppression_rpn(ground_truth, tmp_faceness,
                                                                     box_predict_output,
                                                                     spatial_scale)
            vis_proposal_after_nms(data[train_data.data_name], ground_truth, box_predict_output, nbatch)

    def fit(self, train_data, eval_data=None,
            eval_metric='acc',
            grad_req='write',
            logger=None,
            softmax_metric=None,
            regression_metric=None,
            epoch_end_callback=None):

        # opening files for writing
        f = open("log_rpn.txt", 'w')
        if logger is None:
            logger = logging
        logging.info('Start training with %s', str(self.ctx))
        f.write('Start training with %s\n' % str(self.ctx))

        # arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data=(1, 3, 600, 600), mean_face=(10, 3),
        #                                                              proj_label=(128, 128, 10, 2),
        #                                                              ground_truth=(10, 2), ellipse_label=(10, 5))

        #
        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data=(1, 3, 600, 600), mean_face=(10, 3),
                                                                     proj_label=(128, 128, 10, 2))

        # arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data=(1, 3, 600, 600))


        arg_names = self.symbol.list_arguments()
        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}

        eval_metric = self.network_backprop_setup(grad_req, arg_names, arg_shapes, eval_metric)

        for epoch in range(self.begin_epoch, self.num_epoch):
            if eval_data:
                logger.info(" in eval process...")
                f.write(" in eval process...")

            nbatch = 0
            train_data.reset()
            eval_metric.reset()

            proj_regress_loss_count = .0
            proj_regress_loss_batch = .0

            softmax_count = np.zeros((11, 3))
            softmax_batch = np.zeros((11, 3))

            bbox_predict_loss_t = np.array([.0, .0])
            bbox_predict_loss_b = np.array([.0, .0])

            for data in train_data:

                nbatch += 1
                self.network_input_setup(train_data, data)

                self.executor = self.symbol.bind(self.ctx, self.arg_params,
                                                 args_grad=self.grad_params,
                                                 grad_req=grad_req,
                                                 aux_states=self.aux_params)

                assert len(self.symbol.list_arguments()) == len(self.executor.grad_arrays)
                update_dict = {name: grad_nd for name, grad_nd in
                               zip(self.symbol.list_arguments(), self.executor.grad_arrays) if grad_nd is not None}

                output_dict = {}
                output_buff = {}
                for key, arr in zip(self.symbol.list_outputs(), self.executor.outputs):
                    output_dict[key] = arr
                    output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
                self.executor.forward(is_train=True)
                for key in output_dict:
                    output_dict[key].copyto(output_buff[key])
                self.executor.backward()

                for key, arr in update_dict.items():
                    if key != 'upsample_proposal_weight':
                        self.updater(key, arr, self.arg_params[key])

                softmax_loss_tmp = self.classification_loss_analysis(train_data, data, output_buff, softmax_metric)
                softmax_count += softmax_loss_tmp
                softmax_batch += softmax_loss_tmp

                # a = update_dict['param3d_pred_weight'].asnumpy()
                # print a.max(), a.min()

                proj_regress_tmp = self.proj_regression_loss_analysis(train_data, data, output_buff, regression_metric)
                proj_regress_loss_count += proj_regress_tmp
                proj_regress_loss_batch += proj_regress_tmp

                # ell_label = output_buff["ell_label_output"].asnumpy()
                # bbox_pred = output_buff["ellipse_predict_loss_output"].asnumpy()
                # bbox_predict_tmp = bbox_predict_metric(ell_label, bbox_pred)
                # bbox_predict_loss_t += bbox_predict_tmp
                # bbox_predict_loss_b += bbox_predict_tmp


                # tmp_ellipses = calc_ellipse(bbox_pred, output_buff["box_predict_output"],
                #                             data[train_data.ground_truth_name], spatial_scale=0.5)

                self.executor.outputs[0].wait_to_read()
                self.executor.outputs[1].wait_to_read()
                # self.executor.outputs[2].wait_to_read()
                # self.executor.outputs[3].wait_to_read()

                bbox_predict_tmp = [0,0]
                # proj_regress_tmp = 0
                # softmax_loss_tmp = 0

                self.print_batch_logs(f, data['image_position'], train_data, epoch, nbatch, softmax_loss_tmp, proj_regress_tmp,bbox_predict_tmp)

                if nbatch % 1 == 0:
                    batch_num = epoch * 100000 + nbatch
                    self.visualize_intermediate_outputs(train_data, data, output_buff, batch_num)

                if nbatch % 50 == 0:
                    print_accuracy(softmax_batch, f, train_data.class_names_r, self.bgfg)
                    print "Keypoints projection regression smoothl1 loss:\t", proj_regress_loss_batch / 50
                    f.write("Keypoints projection regression smoothl1 loss:\t%f\n" % (proj_regress_loss_batch / 50))

                    softmax_batch = np.zeros((11, 3))
                    proj_regress_loss_batch = .0

                if nbatch % 500 == 0:
                    if epoch_end_callback != None:
                        epoch_end_callback(epoch * 100000 + nbatch, self.symbol, self.arg_params, self.aux_params)

            name, value = eval_metric.get()
            print_accuracy(softmax_count, f, train_data.class_names_r, self.bgfg)
            logger.info("--->Epoch[%d] Train-cls-%s=%f", epoch, name, value)
            f.write("--->Epoch[%d] Train-cls-%s=%f\n" % (epoch, name, value))

            logger.info("--->Epoch[%d] Train-proj-reg-smoothl1=%f", epoch, proj_regress_loss_count / nbatch)
            f.write("--->Epoch[%d] Train-proj-reg-smoothl1=%f\n" % (epoch, proj_regress_loss_count / nbatch))

        f.close()
