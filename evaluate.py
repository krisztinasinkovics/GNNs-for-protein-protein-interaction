import argparse
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import DataLoader, Data

from sklearn import metrics
from data.dataloader import load_test_data

from core.models import GATN, GCN
from utils.preprocessing import get_node_list, get_edge_lists, get_features_list, get_labels_list, get_graphs_list


def test(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.double()
    model.to(device)
    model.eval()

    labels_list, preds_list = [], []
    for data in loader:
        data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.round()

        labels_list.append(data.y.cpu().numpy())
        preds_list.append(pred.detach().cpu().numpy())

    f1_acc = metrics.f1_score(np.vstack(labels_list), np.vstack(preds_list), average='micro')
    print(f'Test accuracy (F1 score): {f1_acc * 100:.2f}')
    return f1_acc


if __name__ == "__main__":
    # USER INPUT
    parser = argparse.ArgumentParser(
        description='Evaluating GNN node classification for test set of PPI')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory where the unzipped input data is present.')
    parser.add_argument('--model_type', type=str, choices=('GAT', 'GCN'), default='GAT',
                        help='Choose between Graph Attention Network model or Graph Convolution Network model.')
    parser.add_argument('--model_path', type=str, default='saved_models/best_model_GAT.pt',
                        help='Directory where the date is present.')

    args = parser.parse_args()
    input_dir = args.input_dir
    model_type = args.model_type
    model_path = args.model_path

    # Load test data
    test_graph, test_graph_id, test_feats, test_labels = load_test_data(input_dir)

    # Preprocess test data
    list_test_nodes = get_node_list(test_graph, test_graph_id)
    list_test_edge_lists = get_edge_lists(test_graph, test_graph_id)
    list_test_feats = get_features_list(test_feats, test_graph_id)
    list_test_labels = get_labels_list(test_labels, test_graph_id)
    list_test_graphs = get_graphs_list(list_test_nodes, list_test_feats, list_test_labels, list_test_edge_lists)

    test_dataset = []
    for g in list_test_graphs:
        test_dataset.append(torch_geometric.utils.from_networkx(g))

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    if model_type == 'GAT':
        model = GATN(num_features=test_feats.shape[1], num_classes=test_labels.shape[1], hidden_dim=128, num_heads=3)
    elif model_type == 'GCN':
        model = GCN(num_features=test_feats.shape[1], num_classes=test_labels.shape[1], hidden_dim=128)

    model.load_state_dict(torch.load(model_path))
    test(model, test_loader)



