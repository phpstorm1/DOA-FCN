import numpy as np
import math
import os

# deg_predict = tf.placeholder(tf.float32, name='degree_predict')
# deg_label = tf.placeholder(tf.float32, name='degree_label')
# deg_acc, deg_acc_op = tf.metrics.accuracy(labels=tf.argmax(deg_label, 1), predictions=tf.argmax(deg_predict, 1))


def eval_degree_accuracy(predict, label, adjacent_class):
    if predict.shape != label.shape:
        raise Exception('ERROR: the shape of predicted is not the same as of label')
    num_frames, num_class = predict.shape
    if adjacent_class >= num_class:
        raise Exception("ERROR: adjacent_class is greater than the number of class in the input")
    # predict = get_label_from_predict(predict)
    adjacent_predict = sum_adjacent_class(predict, adjacent_class)
    adjacent_label = sum_adjacent_class(label, adjacent_class)

    predict_index = np.argmax(adjacent_predict, axis=1)
    label_index = np.argmax(adjacent_label, axis=1)

    deg_acc = np.sum(np.equal(predict_index, label_index)) / num_frames

    return deg_acc


def sum_adjacent_class(one_hot, adjacent_class):
    num_frames, num_class = one_hot.shape
    num_new_class = math.ceil(num_class / (adjacent_class*2+1))
    sum_adjacent = np.zeros([num_frames, num_new_class])
    for i in range(num_new_class):
        start_sum_idx = i * (adjacent_class*2+1)
        end_sum_idx = num_class if (i+1)*(adjacent_class*2+1) > num_class else (i+1)*(adjacent_class*2+1)
        sum_adjacent[:, i] = np.sum(one_hot[:, start_sum_idx:end_sum_idx], axis=1)
    return sum_adjacent


def eval_adjacent_accuracy(predict, label, adjacent_class):
    if predict.shape != label.shape:
        raise Exception('ERROR: the shape of predicted is not the same as of label')
    num_frames, num_class = predict.shape
    if adjacent_class >= num_class:
        raise Exception("ERROR: adjacent_class is greater than the number of class in the input")
    sum_adjacent_predict = np.zeros(predict.shape)
    for i in range(num_class):
        start_sum_idx = i-adjacent_class if i-adjacent_class >= 0 else 0
        end_sum_idx = i+adjacent_class+1 if i+adjacent_class+1 <= num_class else num_class
        sum_adjacent_predict[:, i] = np.sum(predict[:, start_sum_idx:end_sum_idx], axis=1)
    correct_predict = np.multiply(sum_adjacent_predict, label)
    adj_acc = np.sum(correct_predict) / num_frames
    return adj_acc


def eval_frame_accuracy(predict, label, how_many_previous_frame):
    # averaging through the frames
    if predict.shape != label.shape:
        raise Exception('ERROR: the shape of predicted is not the same as of label')
    num_frames, num_class = predict.shape
    if how_many_previous_frame >= num_frames:
        how_many_previous_frame = num_frames
    # summing adjacent frames
    sum_predict = sum_adjacent_frame(predict, how_many_previous_frame)
    sum_label = sum_adjacent_frame(label, how_many_previous_frame)
    # argmax to get the average
    predict_index = np.argmax(sum_predict, axis=1)
    label_index = np.argmax(sum_label, axis=1)

    frame_acc = np.sum(np.equal(predict_index, label_index)) / num_frames
    return frame_acc


def sum_adjacent_frame(one_hot, how_many_previous_frame):
    num_frames, num_class = one_hot.shape
    sum_previous = np.zeros(one_hot.shape)
    for i in range(num_frames):
        start_sum_idx = 0 if i-how_many_previous_frame<0 else i-how_many_previous_frame
        sum_previous[i, :] = np.sum(one_hot[start_sum_idx:i+1, :], axis=0)
    return sum_previous


def eval_joint_deg_frame(predict, label, adjacent_class, how_many_previous_frame):
    if predict.shape != label.shape:
        raise Exception('ERROR: the shape of predicted is not the same as of label')
    num_frames, num_class = predict.shape
    if adjacent_class >= num_class:
        raise Exception("ERROR: adjacent_class is greater than the number of class in the input")
    if how_many_previous_frame >= num_frames:
        how_many_previous_frame = num_frames
    sum_predict_adj = sum_adjacent_class(predict, adjacent_class)
    sum_label_adj = sum_adjacent_class(label, adjacent_class)

    predict_adj = np.zeros(sum_predict_adj.shape)
    label_adj = np.zeros(sum_label_adj.shape)

    predict_max_col_idx = np.argmax(sum_predict_adj, axis=1)
    label_max_col_idx = np.argmax(sum_label_adj, axis=1)

    for row, col in enumerate(predict_max_col_idx):
        predict_adj[row, col] = 1
    for row, col in enumerate(label_max_col_idx):
        label_adj[row, col] = 1

    return eval_frame_accuracy(predict_adj, label_adj, how_many_previous_frame)


def get_label_from_logits(predict):
    label = np.zeros(predict.shape)
    max_elem = np.argmax(predict, axis=1)
    for row, elem in enumerate(max_elem):
        label[row, elem] = 1
    return label


def get_deg_from_logits(predict, doa_interval, num_doa_class):
    deg_label = np.zeros([predict.shape[0], ])

    max_elem = np.argmax(predict, axis=1)
    min_doa, max_doa = doa_interval
    num_doa = max_doa - min_doa + 1
    doa_per_class = math.ceil(num_doa/num_doa_class)

    for row, elem in enumerate(max_elem):
        deg_label[row] = elem * doa_per_class
    return deg_label


if __name__ == '__main__':
    test_label = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    test_predict = np.array([[0, 0, 0, 1], [0, 1, 0, 0]])
    # print(eval_degree_accuracy(test_predict, test_label, 1))

    logits=np.array([[1, 2, 4], [4, 1, 1]])
    # print(get_label_from_predict(logits))

    # print(test_label[0:1, :])
    # print(np.sum(test_label[0:2, :], axis=0))

    # print(sum_adjacent_frame(test_label, 2))
    # print(np.argmax(sum_adjacent_frame(test_label, 2), axis=1))

    test_label_2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    test_predict_2 = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    # print(eval_frame_accuracy(test_predict_2, test_label_2, 3))

    # test_label_3 = np.array([[0, 0, 0, 1, 0],[1, 0, 0, 0, 0],[0, 0, 1, 0, 0],[1, 0, 0, 0, 0]])
    # print(eval_joint_deg_frame(test_label_3, test_label_3, 1, 3))
    print(eval_adjacent_accuracy(test_predict_2, test_label_2, 3))