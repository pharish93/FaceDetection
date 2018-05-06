import mxnet as mx
import numpy as np

sigma = 1.0

def smoothl1_metric(label, pred, weight):
    rec = 0.0
    size = label.shape[0]
    sigma2 = sigma * sigma
    for i in xrange(0, size):
        # c1 = np.abs(pred[i] - label[i])
        # c2 = c1*weight[i]
        # rec += c2
        rec += np.abs(pred[i] - label[i]) * weight[i]
        '''
        if np.abs(tmp) < sigma2:
            rec += 0.5 * tmp * tmp * sigma2
        else:
            rec += np.abs(tmp) - 0.5 / sigma2
        '''
    if size == 0:
        return 0
    return rec / size


def softmax_metric(label, pred, num_classes):
    pred = np.swapaxes(pred.asnumpy(), 1, 2)
    label = label.asnumpy()
    count = np.zeros((num_classes, 3))
    size = label.shape[1]
    for i in xrange(0, size):
        max_i = np.argmax(pred[0, i, :])
        label_i = int(label[0, i])
        count[label_i, 2] += 1
        if max_i == label_i:
            count[label_i, 0] += 1
            count[label_i, 1] += 1
    return count


def softmax_metric_vis(label, pred, num_classes):
    pred = pred.asnumpy()
    label = label.asnumpy()
    count = np.zeros(3)
    size = pred.shape[0]
    for i in xrange(0, size):
        max_i = np.argmax(pred[i, :])
        count[0] += 1
        if label[i] == 0:
            count[1] += 1
        if max_i == label[i]:
            count[2] += 1
    return count



def get_accuracy(softmax_count, bgfg=False):
    accuracy = 0
    count = 0
    if not bgfg:
        for i in range(0, 11):
            accuracy += softmax_count[i, 0]
            count += softmax_count[i, 2]
    else:
        accuracy = softmax_count[0, 0] + softmax_count[1, 0]
        count = softmax_count[0, 2] + softmax_count[1, 2]

    return accuracy / count


def softmax_metric_vis(label, pred):
    pred = pred.asnumpy()
    print [np.argmax(pred[i, :]) for i in xrange(20)]
    print [int(label[i]) for i in xrange(20)]
    count = np.zeros(3)
    # count[0]: number of instance; count[1]: number of negative instance; count[2]: number of correct
    size = pred.shape[0]
    for i in xrange(0, size):
        max_i = np.argmax(pred[i, :])
        count[0] += 1
        if label[i] == 0:
            count[1] += 1
        if max_i == label[i]:
            count[2] += 1
    return count

def bbox_predict_metric(label, pred):
    res = np.array([.0, .0])
    len = label.shape[0]
    # print label[0]
    # print pred[0]
    for i in xrange(len):
        for j in xrange(4):
            res[0] += np.abs(pred[i, j] - label[i, j])
        res[1] += np.abs(label[i, 4] - pred[i, 4])
    return np.array([res[0] / len / 4, res[1] / len])
