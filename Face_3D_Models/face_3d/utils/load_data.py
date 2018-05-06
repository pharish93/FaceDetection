import numpy as np
from ..logger import logger
from ..config import config
from ..dataset import *



def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path,
                  flip=False):
    """ load ground truth roidb """

    imdb = eval(dataset_name)(image_set_name,root_path,dataset_path)
    roidb , face_points_db , classes , mean_face = imdb.gt_roidb()
    # if flip:
    #     roidb = imdb.append_flipped_images(roidb)
    return roidb, face_points_db, classes, mean_face

