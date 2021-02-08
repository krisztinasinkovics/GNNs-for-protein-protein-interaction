import os
import numpy as np
from utils.preprocessing import read_json_file


def load_train_data(input_dir):
    train_graph_f = os.path.join(input_dir, 'train_graph.json')
    train_graph_id_f = os.path.join(input_dir, 'train_graph_id.npy')
    train_feats_f = os.path.join(input_dir, 'train_feats.npy')
    train_labels_f = os.path.join(input_dir, 'train_labels.npy')

    train_graph = read_json_file(train_graph_f)
    train_graph_id = np.load(train_graph_id_f)
    train_feats = np.load(train_feats_f)
    train_labels = np.load(train_labels_f)

    return train_graph, train_graph_id, train_feats, train_labels


def load_valid_data(input_dir):
    valid_graph_f = os.path.join(input_dir, 'valid_graph.json')
    valid_graph_id_f = os.path.join(input_dir, 'valid_graph_id.npy')
    valid_feats_f = os.path.join(input_dir, 'valid_feats.npy')
    valid_labels_f = os.path.join(input_dir, 'valid_labels.npy')

    valid_graph = read_json_file(valid_graph_f)
    valid_graph_id = np.load(valid_graph_id_f)
    valid_feats = np.load(valid_feats_f)
    valid_labels = np.load(valid_labels_f)

    return valid_graph, valid_graph_id, valid_feats, valid_labels


def load_test_data(input_dir):
    test_graph_f = os.path.join(input_dir, 'test_graph.json')
    test_graph_id_f = os.path.join(input_dir, 'test_graph_id.npy')
    test_feats_f = os.path.join(input_dir, 'test_feats.npy')
    test_labels_f = os.path.join(input_dir, 'test_labels.npy')

    test_graph = read_json_file(test_graph_f)
    test_graph_id = np.load(test_graph_id_f)
    test_feats = np.load(test_feats_f)
    test_labels = np.load(test_labels_f)

    return test_graph, test_graph_id, test_feats, test_labels

