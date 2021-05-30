import matplotlib.pyplot as plt
import numpy as np
import itertools
import os


def imshow_(x, **kwargs):
    if x.ndim == 2:
        plt.imshow(x, interpolation="nearest", **kwargs)
    elif x.ndim == 1:
        plt.imshow(x[:, None].T, interpolation="nearest", **kwargs)
        plt.yticks([])
    plt.axis("tight")


def visualize_pred(predicted, labels, acc, current_video, n_class, dataset):
    vis_path = './visualization/' + dataset + '/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    # concate the prediction and the groundtruth
    pred_norm = predicted / float(n_class - 1)
    labels_norm = labels / float(n_class - 1)
    visual_data = np.vstack((pred_norm, labels_norm))
    # visualization
    plt.figure(figsize=(20, 4))
    imshow_(visual_data, vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("{:.01f}".format(acc))
    plt.savefig(vis_path + str(current_video) + '.png')


# levenstein distance
def levenstein(seq1, seq2, norm=False):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y - 1],
                    matrix[x - 1, y] + 1,
                    matrix[x, y - 1] + 1)
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y - 1] + 1,
                    matrix[x - 1, y] + 1,
                    matrix[x, y - 1] + 1)
    if norm:
        score = (1 - matrix[size_x - 1, size_y - 1] / max(size_x, size_y)) * 100
    else:
        score = matrix[size_x - 1, size_y - 1]

    return score


# get the non-repetitive labels and the start/end point
def get_labels(frame_wise_labels):
    labels = []

    tmp = [0]
    count = 0
    for key, group in itertools.groupby(frame_wise_labels):
        action_len = len(list(group))
        tmp.append(tmp[count] + action_len)
        count += 1
        labels.append(key)
    starts = tmp[:-1]
    ends = tmp[1:]

    return labels, starts, ends


def edit_score(predicted, labels, norm=True):
    seq1, _, _ = get_labels(predicted)
    seq2, _, _ = get_labels(labels)
    return levenstein(seq1, seq2, norm=True)


# calculate the f1 score with differnt threshold(the IoU)
def f_score(predicted, ground_truth, overlap):
    p_label, p_start, p_end = get_labels(predicted)
    y_label, y_start, y_end = get_labels(ground_truth)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)
