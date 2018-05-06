"""
AFLW database
This class loads ground truth notations from AFLW
and transform them into IMDB format. .
"""

import cPickle
import cv2
import os
import numpy as np
import sqlite3

# from PIL import Image, ImageFilter
# from PIL import ImageFile

from ..logger import logger
from imdb_AFLW import IMDB_AFLW
#from pascal_voc_eval import voc_eval
#from ds_utils import unique_boxes, filter_small_boxes


class AFLW(IMDB_AFLW):
    def __init__(self, image_set, root_path, devkit_path):

        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """

        # Harish :: Why do we need to use super ? wouldnt it be better to directly define them is the child class ??
        super(AFLW, self).__init__('AFLW', root_path, devkit_path)  # set self.name

        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = devkit_path
        aflw_db_file = os.path.join(self.root_path,self.data_path, 'aflw.sqlite')
        assert os.path.exists(aflw_db_file), 'Path does not exist: {}'.format(aflw_db_file)
        self.aflw_db_file = aflw_db_file


        self.classes , self.mean_face = self.load_mean_face()
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('%s num_images %d' % (self.name, self.num_images))


        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

        ## Harish Added
        self.AllImg = {}

        self._result = []
        self._feature_points = []
        self._missing_images = 0
        self._oversized_boxes = 0

    def load_mean_face(self):
        con = sqlite3.connect(self.aflw_db_file)
        cursor = con.cursor()
        cursor.execute("SELECT featurecoordtypes.descr,featurecoordtypes.x,featurecoordtypes.y,featurecoordtypes.z \
                        FROM featurecoordtypes")
        result = cursor.fetchall()
        image_class_label = {}
        image_class_label["background"] = 0

        count =1
        for element in result:
            image_class_label[element[0]] = count
            count += 1

        num_class = len(image_class_label)
        mean_face = np.zeros([num_class, 3], np.float32)
        for element in result:
            xyz = np.array([element[1],element[2],element[3]])
            mean_face[image_class_label[element[0]]] = xyz

        return image_class_label,mean_face


    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        con = sqlite3.connect(self.aflw_db_file)
        cursor = con.cursor()
        cursor.execute("SELECT faceimages.image_id FROM faceimages")
        result = cursor.fetchall()
        image_set_index = []
        for element in result:
            image_set_index.extend(list(element))

        return image_set_index

    def image_path_from_index(self, element):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.root_path,self.data_path, 'flickr', element[2])
        if os.path.exists(image_file):
            return image_file
        else :
            self._missing_images +=1
            return -1


    def gt_roidb(self):

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        cache_file_feature_map = os.path.join(self.cache_path, self.name + '_featuremap_roidb.pkl')
        if os.path.exists(cache_file) and os.path.exists(cache_file_feature_map):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('%s gt roidb loaded from %s' % (self.name, cache_file))

            with open(cache_file_feature_map, 'rb') as fid:
                face_points_db = cPickle.load(fid)
            logger.info('%s face point db loaded from %s' % (self.name, cache_file_feature_map))

            return roidb, face_points_db,self.classes,self.mean_face

        aflw_reader = sqlite3.connect(self.aflw_db_file)
        cursor = aflw_reader.cursor()
        cursor.execute(
            "SELECT FaceImages.image_id, FaceRect.face_id, FaceImages.filepath, FaceRect.x, FaceRect.y, FaceRect.w, FaceRect.h, \
                    Faceellipse.x,Faceellipse.y,Faceellipse.ra,Faceellipse.rb,Faceellipse.theta \
                    FROM FaceImages JOIN Faces ON FaceImages.file_id = Faces.file_id \
                    JOIN FaceRect ON FaceRect.face_id = Faces.face_id \
                    JOIN Faceellipse ON Faceellipse.face_id = Faces.face_id")

        self._result = cursor.fetchall()

        cursor.execute(
            "select FeatureCoords.face_id,descr,FeatureCoords.x,FeatureCoords.y \
                    FROM FeatureCoords,FeatureCoordTypes where FeatureCoords.feature_id = FeatureCoordTypes.feature_id ")  # \
                    #JOIN Faces on FaceImages.file_id = Faces.file_id join FaceRect on FaceRect.face_id = Faces.face_id")

        self._feature_points = cursor.fetchall()

        gt_roidb, face_points_db = self.load_aflw_annotations()


        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        logger.info('%s wrote gt roidb to %s' % (self.name, cache_file))

        with open(cache_file_feature_map, 'wb') as fid:
            cPickle.dump(face_points_db, fid, cPickle.HIGHEST_PROTOCOL)
        logger.info('%s wrote feature map roidb to %s' % (self.name, cache_file))


        self.num_images = len(gt_roidb) ## Harish :- Modifying as few images are missing from the data base
        return gt_roidb, face_points_db, self.classes , self.mean_face


    def resize_factor(self,shape,fixed_dim = 600):
        resize_factor = fixed_dim/float(shape[np.argmin(shape[0:2])])
        long_len = max(shape[0] * resize_factor, shape[1] * resize_factor)
        if long_len > 1000:
            resize_factor = 1000.0 / float(shape[np.argmax(shape[0:2])])

        return resize_factor

    def load_bbox(self,position,img_width,img_height):

        x1 = self._result[position][3]
        y1 = self._result[position][4]
        # x2 = self._result[position][3] + self._result[position][5]
        # y2 = self._result[position][4] + self._result[position][6]

        if x1 <= 0:
            x1 = 1
        if y1 <= 0:
            y1 = 1
        # if x2 >= img_width:
        #     x2 = img_width - 1
        # if y2 >= img_height:
        #     y2 = img_height - 1

        ## Harish Jan 27 2018 :- should width, height of box be updated based on box coord  ??
        ## May be :- yes

        width = self._result[position][5]
        height = self._result[position][6]

        # width =  abs(x2 - x1)
        # height = abs(y2 - y1)

        # box = [x1, y1, x2, y2, width, height]
        box = [x1, y1, width, height]
        return box

    def load_ellipse(self,position):

        elp_x = self._result[position][7]
        elp_y = self._result[position][8]
        elp_ra = self._result[position][9]
        elp_rb = self._result[position][10]
        elp_theta = self._result[position][11]

        if elp_theta < 0:
            elp_theta += np.pi

        ellipse = [elp_x, elp_y, elp_ra, elp_rb, elp_theta]
        return ellipse

    def sortby_faceid(self):

        def getKey(element):
            return element[0]
        k = sorted(self._result,key=getKey)
        self._result = k
        k = sorted(self._feature_points,key=getKey)
        self._feature_points = k


    def load_aflw_annotations(self):

        self.sortby_faceid()
        gt_roidb = []

        # Loading image annotations and bounding box information
        i = 0
        while i < len(self._result):
            roi_rec = dict()

            roi_rec['image'] = self.image_path_from_index(self._result[i])
            if roi_rec['image'] != -1:  #if image exists

                size = cv2.imread(roi_rec['image']).shape
                resize_factor = self.resize_factor(size)

                roi_rec['height'] = size[0]
                roi_rec['width'] = size[1]
                roi_rec['resize_factor'] = resize_factor


                num_objs = 0
                while (i+num_objs < len(self._result)) and \
                        (self._result[i+num_objs][0] == self._result[i][0]):   # Check for objects in same image

                    num_objs += 1

                boxes = np.zeros((num_objs, 4), dtype=np.uint16)
                ellipses = np.zeros((num_objs,5),dtype=np.float)
                face_ids = np.zeros((num_objs,1), dtype=np.uint32)

                for ix in range(num_objs):

                    face_ids[ix] = self._result[i+ix][1]
                    boxes[ix, :] = self.load_bbox(i+ix,roi_rec['height'],roi_rec['width'])
                    ellipses[ix, :] = self.load_ellipse(i+ix)

                roi_rec.update({'num_faces': num_objs,
                                'face_ids':face_ids,
                                'boxes': boxes,
                                'ellipses': ellipses,
                                'flipped': False})
                if num_objs != 0:
                    i += num_objs
                else:
                    i += 1      # no objects - move to next line
                gt_roidb.append(roi_rec)

            else:
                i += 1  # image dosen't exist move to next line

        # Loading face box annotations

        face_points_db = dict()
        i = 0
        while i < len(self._feature_points):
            bbox_data = dict()
            num_objs = 0
            while (i + num_objs < len(self._feature_points)) and (
                        self._feature_points[i + num_objs][0] == self._feature_points[i][0]):  # Check for objects in same bpunding box
                num_objs += 1
            for ix in range(num_objs):
                bbox_data[self._feature_points[i + ix][1]] = [self._feature_points[i + ix][2], self._feature_points[i + ix][3]]

            face_points_db[self._feature_points[i][0]] = bbox_data
            if num_objs != 0:
                i += num_objs
            else:
                i += 1

        return gt_roidb,face_points_db
