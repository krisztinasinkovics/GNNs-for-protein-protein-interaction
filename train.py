import os
import numpy as np
import torch
import torch_geometric
from sklearn import metrics
from torch_geometric.data import DataLoader, Data

from core.models import GATN, GCN
from data.dataloader import load_train_data, load_valid_data
from utils.preprocessing import get_node_list, get_edge_lists, get_features_list, get_labels_list, get_graphs_list


def train(model_type, num_features, num_classes, train_loader, val_loader, num_epochs, patience, out_dir):
    if model_type == 'GAT':
        model = GATN(num_features=num_features, num_classes=num_classes, hidden_dim=128, num_heads=3)
    else:
        model = GCN(num_features=num_features, num_classes=num_classes, hidden_dim=128)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    model = model.double()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    patience_counter = 0
    best_train_acc, best_val_acc = 0.0, 0.0

    for epoch in range(num_epochs):
        labels_list, preds_list, loss_list, labels_list_val, preds_list_val = \
            [], [], [], [], []

        model.train()

        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()  # Clear gradients.

            out = model(data.x,
                        data.edge_index)
            loss = criterion(out, data.y.double())
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

            # store train labels, preds and losses
            labels_list.append(data.y.detach().cpu().numpy())
            preds_list.append(out.round().detach().cpu().numpy())
            loss_list.append(loss.detach().cpu().numpy())

        with torch.no_grad():
            model.eval()
            for data in val_loader:
                data.to(device)
                out = model(data.x, data.edge_index)
                val_pred = out.round()

                # store val labels, preds
                labels_list_val.append(data.y.cpu().numpy())
                preds_list_val.append(val_pred.detach().data.cpu().numpy())

        acc_train = metrics.f1_score(np.vstack(labels_list), np.vstack(preds_list), average='micro')
        acc_val = metrics.f1_score(np.vstack(labels_list_val), np.vstack(preds_list_val), average='micro')
        best_val_acc = acc_val if acc_val > best_val_acc else best_val_acc
        best_train_acc = acc_train if acc_train > best_train_acc else best_train_acc

        if epoch % 1 == 0:
            print('*' * 40)
            print(f'Epoch:{epoch}')
            print(f'Train loss:{np.mean(loss_list):.4f}')
            print(f'Train accuracy:{acc_train * 100:.2f}%, Validation accuracy:{acc_val * 100:.2f}%')
            print(f'Best train accuracy:{best_train_acc * 100:.2f}%')
            print(f'Best validation accuracy:{best_val_acc * 100:.2f}%')

        if acc_val < best_val_acc:
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter == patience:
            print('Early stoppling ...')
            break

    model_save_path = os.path.join(out_dir, f'best_model_{model_type}.pt')
    torch.save(model.state_dict(), model_save_path)

    return loss, best_train_acc, best_val_acc


if __name__ == "__main__":
    # USER INPUT
    model_type = 'GAT'  # Graph Attention Network
    # model_type = 'GCN'  # Graph Convolutional Network

    input_dir = '/gdrive/MyDrive/protein_mapping/ppi'
    output_dir = 'saved_models'

    # Load train and valid data
    train_graph, train_graph_id, train_feats, train_labels = load_train_data(input_dir)
    valid_graph, valid_graph_id, valid_feats, valid_labels = load_valid_data(input_dir)

    # Preprocess train data
    list_train_nodes = get_node_list(train_graph, train_graph_id)
    list_train_edge_lists = get_edge_lists(train_graph, train_graph_id)
    list_train_feats = get_features_list(train_feats, train_graph_id)
    list_train_labels = get_labels_list(train_labels, train_graph_id)
    list_train_graphs = get_graphs_list(list_train_nodes, list_train_feats, list_train_labels, list_train_edge_lists)

    train_dataset = []
    for g in list_train_graphs:
        train_dataset.append(torch_geometric.utils.from_networkx(g))

    # Preprocess valid data
    list_valid_nodes = get_node_list(valid_graph, valid_graph_id)
    list_valid_edge_lists = get_edge_lists(valid_graph, valid_graph_id)
    list_valid_feats = get_features_list(valid_feats, valid_graph_id)
    list_valid_labels = get_labels_list(valid_labels, valid_graph_id)
    list_valid_graphs = get_graphs_list(list_valid_nodes, list_valid_feats, list_valid_labels, list_valid_edge_lists)

    valid_dataset = []
    for g in list_valid_graphs:
        valid_dataset.append(torch_geometric.utils.from_networkx(g))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

    train(model_type=model_type,
          num_features=train_feats.shape[1],
          num_classes=train_labels.shape[1],
          train_loader=train_loader,
          val_loader=valid_loader,
          num_epochs=100,
          patience=10,
          out_dir=output_dir)








