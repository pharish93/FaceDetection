import mxnet as mx
import numpy as np
from mxnet.executor_manager import _split_input_slice

import cv2
import math
import random
# from PIL import Image, ImageFilter
from face_3d.symbol.symbol_vgg16 import *
import copy


class position:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0

    def in_box(self, a, b):
        return self.x <= a < self.x + self.width and \
               self.y <= b < self.y + self.height


class Face:
    def __init__(self):
        self.num_keypoint = 0
        self.bbox = position()
        self.proj_bbox = position()
        self.keypoints = {}
        self.ellipse = []


class myImage:
    def __init__(self):
        self.filename = ''
        self.width = 0
        self.height = 0
        self.image = None
        self.faces = {}
        self.num_faces = 0
        self.resize_factor = 0


class FaceIterator(mx.io.DataIter):
    def __init__(self, roi_db, face_points_db, classes, mean_face,
                 rgb_mean=(117, 117, 117),
                 data_name="data",
                 cls_label_name="cls_label",
                 proj_label_name="proj_label",
                 proj_weight_name="proj_weight",
                 ground_truth_name="ground_truth",
                 ellipse_label_name="ellipse_label",
                 bbox_weight_name="bbox_weight",
                 bgfg=False,
                 iterate_range = (0,15000),
                 rpn_symbol=get_vgg_train(),
                 interested_labels=[],
                 num_data_used=-1):

        self.num_data_used = num_data_used
        self.bgfg = bgfg

        self.roi_db = roi_db
        self.face_points_db = face_points_db

        super(FaceIterator, self).__init__()

        self.rgb_mean = np.array(rgb_mean)
        self.mean_face = mean_face

        self.AllImg = {}
        self.data_name = data_name
        self.cls_label_name = cls_label_name

        self.proj_label_name = proj_label_name
        self.proj_weight_name = proj_weight_name

        self.ground_truth_name = ground_truth_name

        self.ellipse_label_name = ellipse_label_name
        self.bbox_weight_name = bbox_weight_name

        self.num_data = len(roi_db)

        self.iterate_range = iterate_range
        self.start_pos = self.iterate_range[0]
        self.stop_pos = self.iterate_range[1]
        self.cursor = self.start_pos - 1
        self.stop = self.stop_pos
        self.data, self.cls_label, self.proj_label, self.proj_weight, self.ground_truth, \
        self.ellipse_label, self.bbox_weight = [], [], [], [], [], [], []
        self.rpn_symbol = rpn_symbol

        self.interested_labels = interested_labels
        self.num_class_name = 'num_classes'
        self.num_class = len(interested_labels) - 1  # removing background as class
        self.class_names = {}
        self.class_names_r = {}
        count = 0
        for element in interested_labels:
            self.class_names[element] = count
            self.class_names_r[count] = element
            count += 1

    def display_input_image(self, img, faces, out_height, out_width):
        img2 = img

        for _, face in faces.iteritems():
            # print 'bbox dimentions : ', (int(face.bbox.x), int(face.bbox.y)),\
            #     int(face.bbox.x + face.bbox.width), int(face.bbox.y + face.bbox.height)
            cv2.rectangle(img2, (int(face.bbox.x), int(face.bbox.y)),
                          (int(face.bbox.x + face.bbox.width), int(face.bbox.y + face.bbox.height)),
                          (0, 255, 0), 3)
            cv2.ellipse(img2, (int(face.ellipse[0]), int(face.ellipse[1])),
                        (int(face.ellipse[2]), int(face.ellipse[3])), int(face.ellipse[4] * 57.2958), 0, 360,
                        (255, 0, 0), 3)

            for k, d in face.keypoints.iteritems():
                cv2.circle(img2, (int(d[0]), int(d[1])), 1, (255, 255, 0), 4)
        img2 = img2[..., ::-1]
        img2 = cv2.resize(img2, (out_width, out_height))
        cv2.imwrite('input_image.jpg', img2)
        # cv2.imshow('image2', img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    @property
    def _read(self):
        self.AllImg[self.cursor] = myImage()
        if self.roi_db[self.cursor]['image'] != -1:

            img, wd_resize, ht_resize = self._read_image_properties()
            num_faces, num_keypoints = self._read_face_properties()

            # Create inputs for feeding into neural network
            #
            # out_height, out_width = self.rpn_symbol.infer_shape(data=(1, 3, ht_resize, wd_resize), mean_face=(10, 3),
            #                                                     proj_label=(ht_resize, wd_resize, 10, 2), ground_truth=(10, 2),
            #                                                     ellipse_label=(10, 5))[1][0][2:4]

            out_height, out_width = self.rpn_symbol.infer_shape(data=(1, 3, ht_resize, wd_resize), mean_face=(10, 3),
                                                                proj_label=(ht_resize, wd_resize, 10, 2))[1][0][2:4]

            # out_height, out_width = self.rpn_symbol.infer_shape(data=(1, 3, ht_resize, wd_resize), mean_face=(10, 3),
            #                                                     proj_label=(ht_resize, wd_resize, 10, 2))[1][0][1:3]
            # out_height, out_width = self.rpn_symbol.infer_shape(data=(1, 3, ht_resize, wd_resize))[1][0][2:4]

            # self.display_input_image(img, self.AllImg[self.cursor].faces,out_height,out_width)

            # Alloting memory
            # confidence map
            cls_label = 255 * np.ones((out_height, out_width), dtype=np.int32)

            # face keypoints projection
            proj_label = np.zeros((out_height, out_width, self.num_class, 2), dtype=np.int32)
            proj_weight = np.zeros((out_height, out_width, self.num_class, 2), dtype=np.float32)

            # gt keypoint locations
            ground_truth = np.zeros((num_keypoints * 9, 2), dtype=np.int32)

            # Bounding box fine tuning
            ellipse_label = np.zeros((num_keypoints * 9, 5), dtype=np.float32)
            bbox_weight = np.zeros((num_keypoints * 9, 5), dtype=np.float32)

            # flagmap for negative anchors creation
            flag_map = np.ones((out_height, out_width), dtype=np.int32)  # used to record negative anchors selection

            ratio_w = float(out_width) / float(wd_resize)
            ratio_h = float(out_height) / float(ht_resize)

            count = 0
            gt_counter = 0

            for i in range(0, num_faces):

                kp_arr = np.zeros((self.num_class, 2), dtype=np.int32)
                kp_warr = np.zeros((self.num_class, 2), dtype=np.float32)

                for kname, kp in self.AllImg[self.cursor].faces[i].keypoints.iteritems():
                    kp_arr[self.class_names[kname] - 1] = np.array([kp[1], kp[0]]) #y ,x
                    kp_warr[self.class_names[kname] - 1] = np.array(
                        [100.0 / self.AllImg[self.cursor].faces[i].bbox.height, # scale wrt height for y , width of box for x
                         100.0 / self.AllImg[self.cursor].faces[i].bbox.width])

                # Iteration over all key points in 1 face :- for each point
                # posx , posy -- > wrt feature map coordinates --> ( 3x3 ) grid for each of the dimensions
                # at each of the location save the class label corresponding to that location, overwritten by most recently used value
                # similarly save the full key point array corresponding to that location

                for kname, kp in self.AllImg[self.cursor].faces[i].keypoints.iteritems():
                    pos_x, pos_y = int(kp[0] * ratio_w), int(kp[1] * ratio_h)

                    # if key points in feature map dimension go out of bound --> Error
                    if pos_x > out_width or pos_y > out_height:
                        print self.AllImg[self.cursor].filename
                        print pos_x, pos_y, out_width, out_height, wd_resize, ht_resize, ratio_w, ratio_h
                    count += 1
                    for idx_y in range(-1, 2):
                        if 0 <= pos_y + idx_y < out_height:
                            for idx_x in range(-1, 2):
                                if 0 <= pos_x + idx_x < out_width:
                                    if self.bgfg:
                                        cls_label[pos_y + idx_y, pos_x + idx_x] = 1
                                    else:
                                        cls_label[pos_y + idx_y, pos_x + idx_x] = self.class_names[kname]
                                    proj_label[pos_y + idx_y, pos_x + idx_x] = kp_arr
                                    proj_weight[pos_y + idx_y, pos_x + idx_x] = kp_warr

                                    ground_truth[gt_counter, :] = np.array([kp[1] + idx_y, kp[0] + idx_x])
                                    ellipse_label[gt_counter, :] = np.array([self.AllImg[self.cursor].faces[i].ellipse[2], #ra
                                                                          self.AllImg[self.cursor].faces[i].ellipse[3], #rb
                                                                          self.AllImg[self.cursor].faces[i].ellipse[1], #y
                                                                          self.AllImg[self.cursor].faces[i].ellipse[0], #x
                                                                          self.AllImg[self.cursor].faces[i].ellipse[4]])
                                    bbox_weight[gt_counter, :] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
                                    gt_counter += 1

            cls_label,flag_map = self._create_negative_data(cls_label,flag_map, count,num_faces,
                                                            out_height,out_width,ratio_h,ratio_w)



            img = np.array(img, dtype=np.float32)
            img = img - self.rgb_mean
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)  # (c,h,w)
            img = np.expand_dims(img, axis=0)  # (1,c,h,w) 1 is batch number

            # flag_map_temp = copy.deepcopy(flag_map)
            # flag_map_temp[flag_map==1]=255
            # cv2.imwrite('flagmap.png',flag_map_temp)
            # cv2.imwrite('cla_lable.png', cls_label)

            cls_label = np.expand_dims(cls_label, axis=0)
            proj_label = np.expand_dims(proj_label, axis=0)
            proj_weight = np.expand_dims(proj_weight, axis=0)

            return [img, cls_label, proj_label, proj_weight, ground_truth[:gt_counter],
                    ellipse_label[:gt_counter],bbox_weight[:gt_counter]]

    def _create_negative_data(self,cls_label,flag_map, count,num_faces,
                                                            out_height,out_width,ratio_h,ratio_w):

        for i in range(0, num_faces):
            bbox = self.AllImg[self.cursor].faces[i].bbox
            y = int(max(0, math.floor(bbox.y * ratio_h)))
            ty = int(min(out_height - 1, math.floor((bbox.y + bbox.height) * ratio_h)))
            x = int(max(0, math.floor(bbox.x * ratio_w)))
            tx = int(min(out_width - 1, math.floor((bbox.x + bbox.width) * ratio_w)))
            flag_map[y: ty, x: tx] = np.zeros((ty - y, tx - x), dtype=np.int32)

        # random choose negative anchors
        # Harish : For each feature point we are putting 9 other background points
        for i in range(0, 9 * count):
            left_anchor = np.nonzero(flag_map)
            if left_anchor[0].size:
                index_neg = random.randint(0, left_anchor[0].size - 1)
                cls_label[left_anchor[0][index_neg], left_anchor[1][index_neg]] = 0
                flag_map[left_anchor[0][index_neg], left_anchor[1][index_neg]] = 0
            else:
                break

        return cls_label,flag_map

    def _read_image_properties(self):

        self.AllImg[self.cursor].filename = self.roi_db[self.cursor]['image']
        wd_resize = int(self.roi_db[self.cursor]['width'] * self.roi_db[self.cursor]['resize_factor'])
        ht_resize = int(self.roi_db[self.cursor]['height'] * self.roi_db[self.cursor]['resize_factor'])
        self.AllImg[self.cursor].width = wd_resize
        self.AllImg[self.cursor].height = ht_resize
        self.AllImg[self.cursor].resize_factor = self.roi_db[self.cursor]['resize_factor']


        img_bgr = cv2.imread(self.AllImg[self.cursor].filename)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if_blur = random.randint(1, 4)
        blur_radius = int(random.choice(range(5, 20, 2)))  # odd numbers

        if if_blur == 1:
            print 'Blured: ', blur_radius
            img = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)

        img = cv2.resize(img, (wd_resize, ht_resize))

        return img, wd_resize, ht_resize

    def _read_face_properties(self):

        num_faces = self.roi_db[self.cursor]['num_faces']
        self.AllImg[self.cursor].num_faces = num_faces

        faces = {}
        num_keypoints = 0
        for j in range(0, num_faces):
            faces[j], num_k = self._read_oneface(j)
            num_keypoints += num_k
        self.AllImg[self.cursor].faces = faces

        return num_faces, num_keypoints

    def _read_oneface(self, face_num):
        oneface = Face()
        resize_factor = self.roi_db[self.cursor]['resize_factor']
        oneface.bbox.x, oneface.bbox.y, oneface.bbox.width, oneface.bbox.height = \
            resize_factor * self.roi_db[self.cursor]['boxes'][face_num]
        ellipse_angle = self.roi_db[self.cursor]['ellipses'][face_num][4]
        oneface.ellipse = resize_factor * self.roi_db[self.cursor]['ellipses'][face_num]
        oneface.ellipse[4]=ellipse_angle
        # Loading Festure points
        temp_kps = self.face_points_db[self.roi_db[self.cursor]['face_ids'][face_num][0]]

        for kname, kp in temp_kps.iteritems():
            if kname in self.interested_labels:
                kp_resized = [pt * resize_factor for pt in kp]
                oneface.keypoints[kname] = kp_resized

        num_kp = len(oneface.keypoints)
        oneface.num_keypoint = num_kp

        return oneface, num_kp

    def get_batch_size(self):
        return 1

    def reset(self):
        self.cursor = self.start_pos - 1

    def iter_next(self):
        self.cursor += 1
        if self.num_data_used != -1 and self.cursor > self.num_data_used:
            return False
        if self.cursor < self.stop_pos and self.num_data:
            return True
        else:
            return False

    def next(self):
        # return a dictionary contains all data needed for one iteration
        if self.iter_next():
            data_image = self._read
            self.data = data_image[0]
            self.cls_label = data_image[1]
            self.proj_label = data_image[2]
            self.proj_weight = data_image[3]
            self.ground_truth = data_image[4]
            self.ellipse_label = data_image[5]
            self.bbox_weight = data_image[6]

            return {self.data_name: self.data,
                    self.cls_label_name: self.cls_label,
                    self.num_class_name: self.num_class,
                    self.proj_label_name: self.proj_label,
                    self.proj_weight_name: self.proj_weight,
                    self.ground_truth_name: self.ground_truth,
                    self.ellipse_label_name:self.ellipse_label,
                    self.bbox_weight_name:self.bbox_weight,
                    'image_position': self.cursor
                    }
        else:
            raise StopIteration
