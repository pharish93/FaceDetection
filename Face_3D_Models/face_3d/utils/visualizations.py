import colorsys
import numpy as np
import cv2




def print_accuracy(softmax_count, f, class_names=None, bgfg=False):
    if not bgfg:
        for i in range(0, 11):
            if softmax_count[i, 2] == 0:
                continue
            print class_names[i], '\t', int(softmax_count[i, 2]), '\t', float(softmax_count[i, 0]) / float(softmax_count[i, 2])
            f.write(
                    "%s\t%d\t%f\n" % (
                        class_names[i], int(softmax_count[i, 2]),
                        float(softmax_count[i, 0]) / float(softmax_count[i, 2])))
    else:
        print 'bg', ':\t', int(softmax_count[0, 2]), '\t', float(softmax_count[0, 0]) / float(softmax_count[0, 2])
        print 'fg', ':\t', int(softmax_count[1, 2]), '\t', float(softmax_count[1, 0]) / float(softmax_count[1, 2])



def get_N_HexCol(N=5):

    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in xrange(N)]
    hex_out = []
    all_rgb = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
        all_rgb.append(rgb)
        # hex_out.append("".join(map(lambda x: chr(x).encode('hex'),rgb)))


    pallete = [[0, 0, 0],
               [128, 0, 0],  # LeftEyeLeftCorner   dark red
               [0, 128, 0],  # RightEyeRightCorner   dark green
               [128, 128, 0],  # LeftEar   dark yellow
               [0, 0, 128],  # NoseLeft    dark blue
               [128, 0, 128],  # NoseRight    dark purple
               [0, 128, 128],  # RightEar    qingse
               [128, 128, 128],  # MouthLeftCorner   gray
               [64, 0, 0],  # MouthRightCorner     darker red
               [255, 0, 0],  # ChinCenter    red
               [0, 0, 255]  # center_between_eyes   blue
               ]
    # return all_rgb
    return pallete

def make_heat_map(image_arr,K,nbatch):
    a = image_arr.shape
    center = get_N_HexCol(K)
    center = np.uint8(center)
    center[0]=[0,0,0]
    res = center[image_arr.flatten()]
    res2 = res.reshape((a[0],a[1],3))

    op_filename = './heatmaps/heatmap' + str(nbatch) + '.png'
    cv2.imwrite(op_filename, res2)

import copy
def vis_proposals(ima_arr,labels,pred,nbatch):
    img = copy.deepcopy(ima_arr[0])
    # img = ima_arr[0]
    img = np.swapaxes(img, 2, 0)
    img = np.swapaxes(img, 1, 0)

    index_nonzero = np.nonzero(labels)
    x = index_nonzero[1]
    y=  index_nonzero[2]
    k = list(set(zip(x, y)))

    for element in k:
        temp = pred[0][element[0]][element[1]]
        x = temp[:,0]
        y = temp[:, 1]
        min_x = min(x[np.nonzero(x)])
        min_y = min(y[np.nonzero(y)])

        max_x = max(temp[:, 0])
        max_y = max(temp[:, 1])

        cv2.rectangle(img, (min_y, min_x), (max_y, max_x), (0, 255, 0), 1)

    op_filename = './proposals/draw_proposals' + str(nbatch) + '.png'
    cv2.imwrite(op_filename, img)

def vis_proposal_after_nms(img_arr,gt,bbox,nbatch):
    img = img_arr[0]
    img = np.swapaxes(img, 2, 0)
    img = np.swapaxes(img, 1, 0)

    for element in gt:

        box = bbox[0][round(element[0] / 2)][round(element[1] / 2)][:]
        cv2.rectangle(img, (box[1], box[0]), (box[1]+box[3], box[0]+box[2]), (0, 255, 0), 1)

    op_filename = './proposals/draw_proposals_afternms' + str(nbatch) + '.png'
    cv2.imwrite(op_filename, img)