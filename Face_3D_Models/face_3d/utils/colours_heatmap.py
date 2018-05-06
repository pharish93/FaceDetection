import colorsys
import numpy as np
import cv2


def get_N_HexCol(N=5):

    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in xrange(N)]
    hex_out = []
    all_rgb = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
        all_rgb.append(rgb)
        # hex_out.append("".join(map(lambda x: chr(x).encode('hex'),rgb)))
    return all_rgb

def make_heat_map(image_arr,K,nbatch):
    a = image_arr.shape
    center = get_N_HexCol(K)
    center = np.uint8(center)
    center[0]=[0,0,0]
    res = center[image_arr.flatten()]
    res2 = res.reshape((a[0],a[1],3))

    op_filename = './heatmaps/heatmap' + str(nbatch) + '.png'
    cv2.imwrite(op_filename, res2)


def vis_proposals(ima_arr,labels,pred,nbatch):
    img = ima_arr[0]
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

        cv2.rectangle(img, (min_y, min_x), (max_y, max_x), (0, 255, 0), 3)

    op_filename = './proposals/draw_proposals' + str(nbatch) + '.png'
    cv2.imwrite(op_filename, img)
