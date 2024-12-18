import yaml
import torch
import random
import warnings
import argparse
import numpy as np
from tqdm import tqdm

from loader import DatasetLoader
from sklearn.cluster import KMeans
from models import HyperEncoder, SimHGCL
from evaluation import linear_evaluation, linear_evaluation_other
warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def train(model_type, num_negs):
    features, hyperedge_index, adjacency_index, processed_hypergraph = data.features, data.hyperedge_index, \
    data.adjacency_index, data.processed_hypergraph
    num_nodes, num_edges = data.num_nodes, data.num_edges
    model.train()
    optimizer.zero_grad()

    # Encoder
    n1, e1 = model(features, hyperedge_index, num_nodes, num_edges)
    n2 = model.forward_gcn(features, adjacency_index)
    
    # for prototype representation of view one
    e1= [torch.mean(n1[index], dim=0, keepdim=True) for index in processed_hypergraph.values()]
    e1 = torch.cat(e1, dim=0)
    
    e2 = [torch.mean(n2[index], dim=0, keepdim=True) for index in processed_hypergraph.values()]    
    e2 = torch.cat(e2, dim=0)

    loss_n = model.node_level_loss(n1, n2, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)
    loss_g = model.group_level_loss(e1, e2, params['tau_g'], batch_size=params['batch_size_1'], num_negs=num_negs)

    loss = loss_n + params['w_g'] * loss_g
    loss.backward()
    optimizer.step()
    return loss.item()


def node_classification_eval(num_splits=20, mode='test'):
    model.eval()
    n, _ = model(data.features, data.hyperedge_index)
    n_g = model.forward_gcn(data.features, data.adjacency_index)

    if data.name == 'pubmed':
        lr = 0.005
        max_epoch = 300
    elif data.name == 'cora' or data.name == 'citeseer':
        lr = 0.005
        max_epoch = 300
    elif data.name == 'Mushroom':
        lr = 0.01
        max_epoch = 200
    else:
        lr = 0.01
        max_epoch = 100

    accs = []
    for i in range(num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(n, data.labels, masks, lr=lr, max_epoch=max_epoch, mode=mode))
    return accs 


def node_clustering_eval():
    model.eval()
    n, _ = model(data.features, data.hyperedge_index)
    
    lab = data.labels.cpu().numpy()
    n_np = n.detach().cpu().numpy()
    
    y_pred = KMeans(n_clusters=len(set(lab))).fit_predict(n_np)
    
    nmi = normalized_mutual_info_score(lab, y_pred)
    ari = adjusted_rand_score(lab, y_pred)
    
    return nmi, ari


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SimHGCL unsupervised learning.')
    parser.add_argument('--dataset', type=str, default='cora', 
        choices=['cora', 'citeseer', 'pubmed', 'cora_coauthor', 'dblp_coauthor', 'house', 'imdb',
      'zoo', 'Mushroom', 'NTU2012', 'ModelNet40', 'dblp_copub', 'aminer', 'cell'])
    parser.add_argument('--model_type', type=str, default='tricl', choices=['tricl_n', 'tricl_ng', 'tricl'])
    parser.add_argument('--wg', type=float, default=1)
    parser.add_argument('--tau_n', type=float, default=0.5)
    parser.add_argument('--tau_g', type=float, default=0.5)
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--eval', type=str, default='classification')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    params = yaml.safe_load(open('config.yaml'))[args.dataset]
    params['tau_n'], params['tau_g'] = args.tau_n, args.tau_g
    params['w_g'] = args.wg
    print(params)

    data = DatasetLoader().load(args.dataset).to(args.device)

    path = './savepoint/'
    accs = []
    for seed in range(args.num_seeds):
        fix_seed(42)
        best_valid_acc = 0
        encoder = HyperEncoder(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
        model = SimHGCL(encoder, params['proj_dim']).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        for epoch in tqdm(range(1, params['epochs'] + 1)):
            loss = train(args.model_type, num_negs=None)
            if (epoch + 1) % 50 == 0:
                valid_acc = np.mean(node_classification_eval(mode='valid'))

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    print(f'\n epoch: {epoch}, valid_acc: {valid_acc:.3f}')
                    torch.save(model.state_dict(), path+args.dataset+'_model.pkl')
        model.load_state_dict(torch.load(path+args.dataset+'_model.pkl'))
        
        acc = node_classification_eval()
        accs.append(acc)
        acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
        print(f'seed: {seed}, test_acc: {acc_mean:.2f}+-{acc_std:.2f}')
