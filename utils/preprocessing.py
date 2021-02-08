import numpy as np
import json
import networkx as nx


def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return js_graph


def get_graph_id_dict(graph_ids):
    """Create a dictionary with keys as group ids in ascending order, values from 0 to num ids"""
    unique_ids = sorted(list(set(graph_ids)))
    return {id: pos for pos, id in enumerate(unique_ids)}


def get_node_list(js_graph, graph_ids):
    """Parse nodes into a list of nodes for each graph"""
    gid_dict = get_graph_id_dict(graph_ids)

    num_graphs = max(graph_ids) - min(graph_ids) + 1
    full_list = [[] for _ in range(num_graphs)]

    for i in range(len(js_graph["nodes"])):
        node = js_graph["nodes"][i]["id"]
        gid = graph_ids[node]
        full_list[gid_dict[gid]].append(node)
    return full_list


def get_edge_lists(js_graph, graph_ids):
    """Parse edges into a list of edge lists for each graph"""
    gid_dict = get_graph_id_dict(graph_ids)

    num_graphs = max(graph_ids) - min(graph_ids) + 1
    full_list = [[] for _ in range(num_graphs)]

    for i in range(len(js_graph["links"])):
        source = js_graph["links"][i]['source']
        target = js_graph["links"][i]['target']
        gid = graph_ids[source]
        full_list[gid_dict[gid]].append([source, target])
    return full_list


def get_edge_indices_from_lists(edge_lists):
    """Convert each elment of list from egde list to edge index"""
    full_edge_index_list = [np.asarray(el).T for el in edge_lists]
    return full_edge_index_list


def get_features_list(features, graph_ids):
    """Get list of features for each graph"""
    feats_list = []
    i = min(graph_ids)
    while i < max(graph_ids):
        a = np.where(graph_ids == i)[0][0]
        b = np.where(graph_ids == i + 1)[0][0]
        feats_list.append(features[a: b, ])
        i += 1

    feats_list.append(features[b:, ])
    return feats_list


def get_labels_list(labels, graph_ids):
    """Get list of labels for each graph"""
    labels_list = []
    i = min(graph_ids)
    while i < max(graph_ids):
        a = np.where(graph_ids == i)[0][0]
        b = np.where(graph_ids == i + 1)[0][0]
        labels_list.append(labels[a: b, ])
        i += 1

    labels_list.append(labels[b:, ])
    return labels_list


def create_digraph(nodes, features, labels, edge_list):
    """Create Directed Graph instance with nodes, it's features and labels as attributes,
       as well as edges"""
    g = nx.DiGraph()
    g.add_edges_from(edge_list)
    for i in range(len(nodes)):
        g.add_node(nodes[i], x=features[i], y=labels[i])
    return g


def get_graphs_list(list_nodes, list_features, list_labels, list_edge_lists):
    """Get a list of directed graphs"""
    graphs = []
    for i in range(len(list_nodes)):
        g_i = create_digraph(list_nodes[i], list_features[i], list_labels[i], list_edge_lists[i])
        graphs.append(g_i)
    return graphs
