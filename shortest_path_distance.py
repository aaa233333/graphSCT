import networkx as nx
import numpy as np
import multiprocessing as mp
import random
import torch
def cal_shortest_path_distance(self, adj, approximate,idx_train_set_class):

    n_nodes = self.num_nodes
    Adj = adj.detach().cpu().numpy()  #adj.to_dense().numpy()
    G = nx.from_numpy_array(Adj)
    G.edges(data=True)#返回图 G 中所有边的数据，包括源节点、目标节点以及边的属性。
    dists_array = np.zeros((n_nodes, n_nodes))
    dists_dict = all_pairs_shortest_path_length_parallel(G, cutoff=approximate if approximate > 0 else None)

    cnt_disconnected = 0

    for i, node_i in enumerate(G.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(G.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist == -1:
                cnt_disconnected += 1
            if dist != -1:
                dists_array[node_i, node_j] = dist


    return dists_array

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)   # unweighted
    return dists_dict


def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=4):
    nodes = list(graph.nodes)#根据图的节点数量来调整并行工作进程数
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=num_workers)#num_workers 是指定的并行工作进程数
    results = [pool.apply_async(single_source_shortest_path_length_range,
                                args=(graph, nodes[int(len(nodes) / num_workers * i):int(len(nodes) / num_workers * (i + 1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict #返回的是一个字典


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def shortest_path_distance_ht(adj,idx_train_set_class, idx_train,labels,approximate, above_head=None, below_tail=None, below=None):
    idx_train_set = {}
    idx_train_set['HH'] = []
    idx_train_set['HT'] = []
    idx_train_set['TH'] = []
    idx_train_set['TT'] = []
    n_nodes = len(labels)
    Adj = adj.to_dense().numpy()  # adj.to_dense().numpy()
    G = nx.from_numpy_array(Adj)
    G.edges(data=True)  # 返回图 G 中所有边的数据，包括源节点、目标节点以及边的属性。
    dists_array = np.zeros((n_nodes, n_nodes))
    dists_dict = all_pairs_shortest_path_length_parallel(G, cutoff=approximate if approximate > 0 else None)#返回一个字典

    cnt_disconnected = 0
    #节点之间的最短路径长度存储在一个二维数组 dists_array 中

    for i, node_i in enumerate(G.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(G.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist == -1:
                cnt_disconnected += 1
            if dist != -1:
                dists_array[node_i, node_j] = dist

    avg_dists = np.zeros(shape=(n_nodes,), dtype=float)  # 创建大小为 n_nodes * 1 的空数组
    for iter in range(n_nodes):
        total_dist = np.sum(dists_array[iter, :])#-np.sum(dists_array[iter, idx_train])  # 计算未标记节点到该节点的最短路径长度总和
        avg_dist = total_dist / (n_nodes - 1)  # -len(idx_train) 计算平均最短路径长度
        avg_dists[iter] = avg_dist  # 存储平均最短路径长度
    #avg_dists=np.squeeze(avg_dists)

    '''
    avg_dists=np.zeros(shape=(n_nodes,), dtype=float)
    unlabeled_nodes=torch.tensor(list(set(torch.arange(len(labels)).tolist()) - set(idx_train.tolist())))
    for unlabeled_node in unlabeled_nodes:
        distances = []
        for labeled_node in idx_train:
            if unlabeled_node != labeled_node:
                distance = dists_array[unlabeled_node, labeled_node]  # 获取最短路径距离
                distances.append(distance)
            avg_dists[labeled_node]=np.mean(distances)
    '''
    above_head = {}
    below_tail = {}
    degree_dict = {}

    for sep in ['H', 'T']:
        if len(idx_train_set_class[sep]) == 0:
            continue

        elif len(idx_train_set_class[sep]) == 1:
            idx = idx_train_set_class[sep]
            if sep == 'H':
                rand = random.choice(['HH', 'HT'])
                idx_train_set[rand].append(int(idx))
            elif sep == 'T':
                rand = random.choice(['TH', 'TT'])
                idx_train_set[rand].append(int(idx))

        else:
            dist_idx_train = avg_dists[idx_train_set_class[sep]]

            above_head = below + 1
            below_tail = below
            gap_head = abs(dist_idx_train - (below + 1))
            gap_tail = abs(dist_idx_train - below)

            if sep == 'H':
                idx_train_set['HH'] = list(map(int, idx_train_set_class[sep][gap_head < gap_tail]))
                idx_train_set['HT'] = list(map(int, idx_train_set_class[sep][gap_tail < gap_head]))

                if sum(gap_head == gap_tail) > 0:
                    for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                        rand = random.choice(['HH', 'HT'])
                        idx_train_set[rand].append(int(idx))

            elif sep == 'T':
                idx_train_set['TH'] = list(map(int, idx_train_set_class[sep][gap_head < gap_tail]))
                idx_train_set['TT'] = list(map(int, idx_train_set_class[sep][gap_tail < gap_head]))

                if sum(gap_head == gap_tail) > 0:
                    for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                        rand = random.choice(['TH', 'TT'])
                        idx_train_set[rand].append(int(idx))

    for idx in ['HH', 'HT', 'TH', 'TT']:
        random.shuffle(idx_train_set[idx])
        idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

    return idx_train_set, degree_dict, avg_dists, above_head, below_tail