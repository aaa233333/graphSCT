import numpy as np
import torch
import random
from sklearn.metrics import f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
import os.path as osp
import os
import logging
import sys
from torch_scatter import scatter_add
import torch.nn.functional as F
# LT dataset from GraphENS
def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    # Check whether inv_indices is correct
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(
        len(n_data))).sum().abs() < 1e-12

    mu = np.power(1 / ratio, 1 / (n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        # Check whether the number of class is greater than or equal to 1
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        if i < 1:
            n_round.append(1)
        else:
            n_round.append(10)

    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]
    # print(class_num_list);input()

    remove_class_num_list = [n_data[i].item() - class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) * original_mask])

    for i in indices.numpy():
        for r in range(1, n_round[i] + 1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list, [])] = False

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = (row_mask * col_mask).type(torch.bool)

            # Compute degree
            degree = scatter_add(torch.ones_like(row[edge_mask]), row[edge_mask]).to(row.device)
            if len(degree) < len(label):
                degree = torch.cat([degree, degree.new_zeros(len(label) - len(degree))], dim=0)
            degree = degree[cls_idx_list[i]]

            # Remove nodes with low degree first (number increases as round increases)
            _, remove_idx = torch.topk(degree, (r * remove_class_num_list[i]) // n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.numpy())

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list, [])] = False

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = (row_mask * col_mask).type(torch.bool)

    train_mask = (node_mask * train_mask).type(torch.bool)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) * train_mask]
        idx_info.append(cls_indices)
    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask

def split_manual(labels, c_train_num, idx_map):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    class_num_list = []
    idx_info = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25 
    c_num_mat[:,2] = 55 

    for i in range(num_classes):
        idx = list(idx_map.keys())[list(idx_map.values()).index(i)]
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('OG:{:d} -> NEW:{:d}-th class sample number: {:d}'.format(idx, i, len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        c_num_mat[i,0] = c_train_num[i]
        # =============
        cls_indices = c_idx[:c_num_mat[i, 0]]
        idx_info.append(torch.tensor(cls_indices))
        class_num_list.append(c_num_mat[i, 0])
        # =================
        val_idx = val_idx + c_idx[c_train_num[i]: c_train_num[i]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_train_num[i]+c_num_mat[i,1]: c_train_num[i]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    # =============
    # idx_info=torch.tensor(idx_info)

    stats = labels[train_idx]
    n_data = []
    for i in range(num_classes):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(num_classes, dtype=np.int64)
    for i in range(num_classes):
        inv_indices[indices[i].item()] = i
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    # =================
    return train_idx, val_idx, test_idx, c_num_mat,list(class_num_list),idx_info

def split_manual_lt(labels, idx_train, idx_val, idx_test):
    num_classes = len(set(labels.tolist()))
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25
    c_num_mat[:,2] = 55

    for i in range(num_classes):
        c_idx = (labels[idx_train]==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        val_lists = list(map(int,idx_val[labels[idx_val]==i]))
        test_lists = list(map(int,idx_test[labels[idx_test]==i]))
        random.shuffle(val_lists)
        random.shuffle(test_lists)

        c_num_mat[i,0] = len(c_idx)

        val_idx = val_idx + val_lists[:c_num_mat[i,1]]
        test_idx = test_idx + test_lists[:c_num_mat[i,2]]

    train_idx = torch.LongTensor(idx_train)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat

def split_natural(labels, idx_map):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    class_num_list = []
    idx_info = []

    for i in range(num_classes):
        idx = list(idx_map.keys())[list(idx_map.values()).index(i)]
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('OG:{:d} -> NEW:{:d}-th class sample number: {:d}'.format(idx, i, len(c_idx)))
        c_num = len(c_idx)

        if c_num == 3:
            c_num_mat[i, 0] = 1
            c_num_mat[i, 1] = 1
            c_num_mat[i, 2] = 1
        else:
            random.shuffle(c_idx)
            c_idxs.append(c_idx)
            c_num_mat[i,0] = int(c_num*0.1) # 10% for train
            c_num_mat[i,1] = int(c_num*0.1) # 10% for validation
            c_num_mat[i,2] = int(c_num*0.8) # 80% for test
        # print('[{}-th class] Total: {} | Train: {} | Val: {} | Test: {}'.format(i,len(c_idx), c_num_mat[i,0], c_num_mat[i,1], c_num_mat[i,2]))

        train_idx = train_idx + c_idx[:c_num_mat[i,0]]
        #=============
        cls_indices=c_idx[:c_num_mat[i,0]]
        idx_info.append(torch.tensor(cls_indices))
        class_num_list.append(c_num_mat[i, 0])
        #=================
        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    # =============
    #idx_info=torch.tensor(idx_info)

    stats = labels[train_idx]
    n_data = []
    for i in range(num_classes):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(num_classes, dtype=np.int64)
    for i in range(num_classes):
        inv_indices[indices[i].item()] = i
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    # =================
    return train_idx, val_idx, test_idx, c_num_mat,list(class_num_list),idx_info

def split_amazon(labels, idx_train, idx_val, idx_test):
    num_classes = len(set(labels.tolist()))
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)
    for i in range(num_classes):
        c_idx = (labels[idx_train] == i).nonzero()[:, -1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        val_lists = list(map(int, idx_val[labels[idx_val] == i]))
        test_lists = list(map(int, idx_test[labels[idx_test] == i]))
        random.shuffle(val_lists)
        random.shuffle(test_lists)

        c_num_mat[i, 0] = len(c_idx)
        c_num_mat[i, 1] = len(val_lists)
        c_num_mat[i, 2] = len(test_lists)

        val_idx = val_idx + val_lists[:c_num_mat[i, 1]]
        test_idx = test_idx + test_lists[:c_num_mat[i, 2]]
    train_idx = torch.LongTensor(idx_train)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat

def separate_class_degree(adj, idx_train_set_class, above_head=None, below_tail=None, below=None, rand=False, is_eval=False):
    idx_train_set = {}
    idx_train_set['HH'] = []
    idx_train_set['HT'] = []
    idx_train_set['TH'] = []
    idx_train_set['TT'] = []

    adj_dense = adj.to_dense()
    adj_dense[adj_dense != 0] = 1
    degrees = np.array(list(map(int, torch.sum(adj_dense, dim=0))))

    if rand:
        for sep in ['H', 'T']:
            idxs = np.array(idx_train_set_class[sep])
            np.random.shuffle(idxs)
        
            idx_train_set[sep+'H'] = idxs[:int(len(idxs)/2)]
            idx_train_set[sep+'T'] = idxs[int(len(idxs)/2):]
            
        for idx in ['HH', 'HT', 'TH', 'TT']:
            random.shuffle(idx_train_set[idx])
            idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

            degree_dict = {}
            above_head = 0
            below_tail = 0
        
        return idx_train_set, degree_dict, degrees, above_head, below_tail


    if not is_eval:
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
                degrees_idx_train = degrees[idx_train_set_class[sep]]

                above_head = below + 1
                below_tail = below
                gap_head = abs(degrees_idx_train - (below+1))
                gap_tail = abs(degrees_idx_train - below)

                if sep == 'H':
                    idx_train_set['HH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                    idx_train_set['HT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                    if sum(gap_head == gap_tail) > 0:
                        for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                            rand = random.choice(['HH', 'HT'])
                            idx_train_set[rand].append(int(idx))

                elif sep == 'T':
                    idx_train_set['TH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                    idx_train_set['TT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                    if sum(gap_head == gap_tail) > 0:
                        for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                            rand = random.choice(['TH', 'TT'])
                            idx_train_set[rand].append(int(idx))

        for idx in ['HH', 'HT', 'TH', 'TT']:
            random.shuffle(idx_train_set[idx])
            idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

        return idx_train_set, degree_dict, degrees, above_head, below_tail
    
    elif is_eval:
        for sep in ['H', 'T']:
            if len(idx_train_set_class[sep]) == 0:
                continue

            else:
                degrees_idx_train = degrees[idx_train_set_class[sep]]

                gap_head = abs(degrees_idx_train - above_head)
                gap_tail = abs(degrees_idx_train - below_tail)

                if sep == 'H':
                    if len(idx_train_set_class[sep]) == 1:
                        if gap_head < gap_tail:
                            idx_train_set['HH'].append(int(idx_train_set_class[sep]))
                        elif gap_tail < gap_head:
                            idx_train_set['HT'].append((idx_train_set_class[sep]))
                        else:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['HH', 'HT'])
                                idx_train_set[rand].append(int(idx))
                    else:
                        idx_train_set['HH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                        idx_train_set['HT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                        if sum(gap_head == gap_tail) > 0:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['HH', 'HT'])
                                idx_train_set[rand].append(int(idx))

                elif sep == 'T':
                    if len(idx_train_set_class[sep]) == 1:
                        if gap_head < gap_tail:
                            idx_train_set['TH'].append(int(idx_train_set_class[sep]))
                        elif gap_tail < gap_head:
                            idx_train_set['TT'].append(int(idx_train_set_class[sep]))
                        else:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['TH', 'TT'])
                                idx_train_set[rand].append(int(idx))
                    else:
                        idx_train_set['TH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                        idx_train_set['TT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                        if sum(gap_head == gap_tail) > 0:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['TH', 'TT'])
                                idx_train_set[rand].append(int(idx))
            
        for idx in ['HH', 'HT', 'TH', 'TT']:
            idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])
                
        return idx_train_set

def separate_eval(idx_eval, labels, ht_dict_class, degrees, above_head, below_tail):
    idx_eval_set = {}
    idx_eval_set['HH'] = []
    idx_eval_set['HT'] = []
    idx_eval_set['TH'] = []
    idx_eval_set['TT'] = []
    
    for idx in idx_eval:
        label = int(labels[idx])
        degree = int(degrees[idx])
        if (label in ht_dict_class['H']) and (degree >= above_head):
            idx_eval_set['HH'].append(int(idx))

        elif (label in ht_dict_class['H']) and (degree <= below_tail):
            idx_eval_set['HT'].append(int(idx))

        elif (label in ht_dict_class['T']) and (degree >= above_head):
            idx_eval_set['TH'].append(int(idx))

        elif (label in ht_dict_class['T']) and (degree <= below_tail):
            idx_eval_set['TT'].append(int(idx))
        
    
    for idx in ['HH', 'HT', 'TH', 'TT']:
        random.shuffle(idx_eval_set[idx])
        idx_eval_set[idx] = torch.LongTensor(idx_eval_set[idx])
            
    return idx_eval_set

def separate_ht(samples_per_label, labels, idx_train, method='pareto_28', rand=False, manual=False):
    class_dict = {}
    idx_train_set = {}

    if rand:
        ht_dict = {}
        arr = np.array(idx_train)
        np.random.shuffle(arr)
        sample_num = int(idx_train.shape[0]/2)
        sample_label_num = int(len(labels.unique())/2)
        label_list = np.array(labels.unique())
        np.random.shuffle(label_list)
        ht_dict['H'] = label_list[0:sample_label_num]
        ht_dict['T'] = label_list[sample_label_num:]

        idx_train_set['H'] = arr[0:sample_num]
        idx_train_set['T'] = arr[sample_num:]

    elif manual:
        ht_dict = {}
        samples = samples_per_label
        point = np.arange(len(samples_per_label)-1)[list(map(lambda x: samples[x] != samples[x+1], range(len(samples)-1)))][0]
        label_list = np.array(labels.unique())
        ht_dict['H'] = label_list[0:point+1]
        ht_dict['T'] = label_list[point+1:]

        print('Samples per label:', samples_per_label)
        print('Separation:', ht_dict.items())

        idx_train_set['H'] = []
        idx_train_set['T'] = []
        for label in label_list:
            idx = 'H' if label <= point else 'T'
            idx_train_set[idx].extend(torch.LongTensor(idx_train[labels[idx_train] == label]))
            
    else:
        ht_dict = separator_ht(samples_per_label, method) # H/T

        print('Samples per label:', samples_per_label)
        print('Separation:', ht_dict.items())

        for idx, value in ht_dict.items():
            class_dict[idx] = []
            idx_train_set[idx] = []
            idx = idx
            label_list = value

            for label in label_list:
                class_dict[idx].append(label)
                idx_train_set[idx].extend(torch.LongTensor(idx_train[labels[idx_train] == label]))
            
    for idx in list(ht_dict.keys()):
        random.shuffle(idx_train_set[idx])
        idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

    return idx_train_set, ht_dict


def separator_ht(dist, method='pareto_28', degree=False): # Head / Tail separator
    head = int(method[-2]) # 2 in pareto_28
    tail = int(method[-1]) # 8 in pareto_28
    head_idx = int(len(dist) * (head/10))
    ht_dict = {}

    if head_idx == 0:
        ht_dict['H'] = list(range(0, 1))
        ht_dict['T'] = list(range(1, len(dist)))
        return ht_dict

    else:
        crierion = dist[head_idx].item()

        case1_h = sum(np.array(dist) >= crierion)
        case1_t = sum(np.array(dist) < crierion)

        case2_h = sum(np.array(dist) > crierion)
        case2_t = sum(np.array(dist) <= crierion)

        gap_case1 = abs(case1_h/case1_t - head/tail)
        gap_case2 = abs(case2_h/case2_t - head/tail)

        if gap_case1 < gap_case2:
            idx = sum(np.array(dist) >= crierion)
            ht_dict['H'] = list(range(0, idx))
            ht_dict['T'] = list(range(idx, len(dist)))

        elif gap_case1 > gap_case2:
            idx = sum(np.array(dist) > crierion)
            ht_dict['H'] = list(range(0, idx))
            ht_dict['T'] = list(range(idx, len(dist)))

        else:
            rand = random.choice([1, 2])
            if rand == 1:
                idx = sum(np.array(dist) >= crierion)
                ht_dict['H'] = list(range(0, idx))
                ht_dict['T'] = list(range(idx, len(dist)))
            else:
                idx = sum(np.array(dist) > crierion)
                ht_dict['H'] = list(range(0, idx))
                ht_dict['T'] = list(range(idx, len(dist)))

        return ht_dict

def accuracy(output, labels, sep_point=None, sep=None, pre=None):

    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point # [4,5,6] -> [0,1,2]

    if output.shape != labels.shape:
        if len(labels) == 0:
            return np.nan
        preds = output.max(1)[1].type_as(labels)
    else:
        preds= output

    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def classification(output, labels, sep_point=None, sep=None):
    target_names = []
    if len(labels) == 0:
        return np.nan
    else:
        if sep in ['T', 'TH', 'TT']:
            labels = labels - sep_point
        pred = output.max(1)[1].type_as(labels)
        for i in labels.unique():
            target_names.append(f'class_{int(i)}')

        return classification_report(labels, pred)

def confusion(output, labels, sep_point=None, sep=None):
    if len(labels) == 0:
        return np.nan
    else:
        if sep in ['T', 'TH', 'TT']:
            labels = labels - sep_point
        
        pred = output.max(1)[1].type_as(labels)
    
        return confusion_matrix(labels, pred)

def performance_measure(output, labels, sep_point=None, sep=None, pre=None):
    acc = accuracy(output, labels, sep_point=sep_point, sep=sep, pre=pre)*100

    if len(labels) == 0:
        return np.nan
    
    if output.shape != labels.shape:
        output = torch.argmax(output, dim=-1)

    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point # [4,5,6] -> [0,1,2]

    macro_F = f1_score(labels.cpu().detach(), output.cpu().detach(), average='macro')*100
    gmean = geometric_mean_score(labels.cpu().detach(), output.cpu().detach(), average='macro')*100
    bacc = balanced_accuracy_score(labels.cpu().detach(), output.cpu().detach())*100

    return acc, macro_F, gmean, bacc

def adj_mse_loss(adj_rec, adj_tgt=None, adj_mask = None):
    adj_tgt[adj_tgt != 0] = 1

    edge_num = adj_tgt.nonzero().shape[0] #number of non-zero
    total_num = adj_tgt.shape[0]**2 #possible edge

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2) # element-wise

    return loss

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)

def refine_label_order(labels):
    print('Refine label order, Many to Few')
    num_labels = labels.max() + 1
    num_labels_each_class = np.array([(labels == i).sum().item() for i in range(num_labels)])
    sorted_index = np.argsort(num_labels_each_class)[::-1]
    idx_map = {sorted_index[i]:i for i in range(num_labels)}
    new_labels = np.vectorize(idx_map.get)(labels.numpy())

    return labels.new(new_labels), idx_map

def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[idx], dim=1))
    return sum_m 

def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    deg = torch.sum(adj.to_dense(), dim=1)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt).to_sparse()
    adj = torch.spmm(deg_inv_sqrt, adj.to_dense()).to_sparse()
    
    return adj

def normalize_sym(adj):
    """Symmetric-normalize sparse matrix"""
    deg = torch.sum(adj.to_dense(), dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt).to_sparse()

    adj = torch.spmm(deg_inv_sqrt, adj.to_dense()).to_sparse()
    adj = torch.spmm(adj, deg_inv_sqrt.to_dense()).to_sparse()

    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def scheduler(epoch, curriculum_ep=500, func='convex'):
    if func == 'convex':
        return np.cos((epoch * np.pi) / (curriculum_ep * 2))
    elif func == 'concave':
        return np.power(0.99, epoch)
    elif func == 'linear':
        return 1 - (epoch / curriculum_ep)
    elif func == 'composite':
        return (1/2) * np.cos((epoch*np.pi) / curriculum_ep) + 1/2

def setupt_logger(save_dir, text, filename = 'log.txt'):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(text)
    # for each in logger.handlers:
    #     logger.removeHandler(each)
    logger.setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info("======================================================================================")
    return logger

def set_filename(args):
    rec_with_ep_pre = 'True_ep_pre_' + str(args.ep_pre) + '_rw_' + str(args.rw) if args.rec else 'False'

    if args.im_ratio == 1: # Natural Setting
        results_path = f'./results/natural/{args.dataset}'
        logs_path = f'./logs/natural/{args.dataset}'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        textname = f'cls_og_{args.cls_og}_rec_{rec_with_ep_pre}_cw_{args.class_weight}_gamma_{args.gamma}_alpha_{args.alpha}_sep_class_{args.sep_class}_degree_{args.sep_degree}_cur_ep_{args.curriculum_ep}_lr_{args.lr}_{args.lr_expert}_dropout_{args.dropout}.txt'
        text = open(f'./results/natural/{args.dataset}/({args.layer}){textname}', 'w')
        file = f'./logs/natural/{args.dataset}/({args.layer})lte4g.txt'
        
    else: # Manual Imbalance Setting (0.2, 0.1, 0.05)
        results_path = f'./results/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}'
        logs_path = f'./logs/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        textname = f'cls_og_{args.cls_og}_rec_{rec_with_ep_pre}_cw_{args.class_weight}_gamma_{args.gamma}_alpha_{args.alpha}_sep_class_{args.sep_class}_degree_{args.sep_degree}_cur_ep_{args.curriculum_ep}_lr_{args.lr}_{args.lr_expert}_dropout_{args.dropout}.txt'
        text = open(f'./results/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}/({args.layer}){textname}', 'w')
        file = f'./logs/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}/({args.layer})lte4g.txt'
        
    return text, file

def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info

def remove_ht(class_num_list, prev_out_local, idx_info, idx_train,idx_train_sep,labels,tau=2,ht_dict=None):
    class_num_list = torch.tensor(class_num_list)
    label_list = max(ht_dict['H'])
    #class_num_list=class_num_list[:label_list]
    sorted_n_data, indices = torch.sort(class_num_list, descending=True)
    n_cls=len(class_num_list)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(class_num_list))[indices][torch.tensor(inv_indices)] - torch.arange(
        len(class_num_list))).sum().abs() < 1e-12


    class_num_list_idx = []
    idx_info_=[]
    for i in range(n_cls):
        data_num=(labels[idx_train_sep]==i).sum().item()
        class_num_list_idx.append(int(data_num))

        cls_indices=set(idx_info[i].tolist())&set(idx_train_sep.tolist())#
        #cls_indices=np.intersect1d(idx_info[i].cpu().numpy(),idx_train_sep.cpu().numpy())
        idx_info_.append(cls_indices)
    class_num_list_idx = np.array(class_num_list_idx)
    class_num_list_idx = class_num_list_idx[inv_indices]
    # 将集合转换为列表
    idx_info_sep = [list(s) for s in idx_info_]
    #idx_info_sep=torch.tensor(list_of_sets, dtype=torch.int64)
    #idx_info_sep=torch.tensor(idx_info_sep)

    max_num, n_cls_idx = max(class_num_list_idx), len(class_num_list_idx)
    if label_list==0:
        max_num = sum(class_num_list_idx)
    else:
        max_num = sum(class_num_list_idx) / label_list
    sampling_list =torch.tensor(class_num_list_idx) - max_num * torch.ones(n_cls_idx )

    prev_out_local = F.softmax(prev_out_local / tau, dim=1)
    prev_out_local = prev_out_local.cpu()

    idx_train_sep_list=idx_train_sep.cpu().tolist()
    local2global = {i: idx_train_sep_list[i] for i in range(len(idx_train_sep_list))}
    global2local = dict([val, key] for key, val in local2global.items())
    idx_info_list = [item for item in idx_info_sep ]
    idx_info_local = [torch.tensor(list(map(global2local.get, k))) for k in idx_info_list]
    idx_train_remove=[]

    #remove_idx_list = [[] for _ in range(n_cls)]
    for cls_idx, num in enumerate(sampling_list):
        num = int(num.item())
        if num <= 0:
            continue

        # first sampling
        prob = 1 - prev_out_local[idx_info_local[cls_idx]][:, cls_idx].squeeze()
        src_idx_local = torch.multinomial(prob + 1e-12, num,replacement=True)
        src_idx = idx_train[idx_info_local[cls_idx][src_idx_local]]

        idx_train_remove.append(src_idx)

    if len(idx_train_remove)>0:
        idx_train_remove=torch.cat(idx_train_remove)
        idx_train_remove=torch.unique(idx_train_remove)

        idx_train = torch.tensor(list(set(idx_train.tolist()) - set(idx_train_remove.tolist())))
        #idx_train_new=np.delete(idx_train.cpu().numpy(),idx_train_remove.cpu().numpy())
        #idx_train_new=torch.from_numpy(idx_train_new)
        return idx_train
    else:

        return idx_train



def adasyn_ht(embed,class_num_list, labels, idx_info, idx_train,adj=None,ht_dict=None, beta=1, K=5):

    adj_new=None
    idx_train_append = torch.Tensor()
    idx_train_append = idx_train_append.to(embed.device)
    max_num=int(max(class_num_list))
    label_list=ht_dict['T']
    for label in label_list: #cora的话label=5 6是T
        chosen=idx_train[labels[idx_train] == label]
        G = (max_num - chosen.shape[0]) * beta
        ratio = []
        idx_neighbor=[]
        chosen_embed = embed[idx_train, :]
        #chosen_labels = labels[idx_train]
        p = np.sum(np.square(chosen_embed.detach().cpu().numpy()), axis=1)
        distance = -2 * np.dot(chosen_embed.detach().cpu().numpy(), chosen_embed.detach().cpu().numpy().T) + p.reshape(
            1, -1) + p.reshape(-1, 1)


        for index,d in enumerate(idx_train):
            if labels[index]==label:
                ner_index = distance[index].argsort()[1:K + 1]  #
                label_ner = labels[distance[index].argsort()[1:K + 1]]
                r = (K-label_ner[label_ner == label].shape[0] )/ K
                ratio.append([index, r,ner_index])
            else:
                continue
        r = [ri[1] for ri in ratio]
        ratio_sum = sum(r)
        if ratio_sum == 0:
            continue
            #print('data is easy to classify! No necessary to do ADASYN')
            #return embed, labels, idx_train, adj.detach()

        g = [round(ri[1] / ratio_sum * G) for ri in ratio]#round四舍五入
        new_embed=[]
        chosen_index=[]

        for index1, info in enumerate(ratio):
            minority_point_index = info[0]
            x_i = embed[minority_point_index]
            #x_i=x_i.cpu().detach()
            ner = cal_knn(torch.unsqueeze(x_i, 0), embed[labels == label], K)[0]
            for j in range(0, g[index1]):
                random_index = np.random.choice(ner.shape[1])
                la = np.random.ranf(1)
                idx_neighbor.append(ner[:, random_index])
                x_zi = embed[labels == label][ner[:, random_index]]
                #x_zi=torch.squeeze(x_zi)
                generate_data = x_i.detach().cpu().numpy() + (x_zi.detach().cpu().numpy()- x_i.detach().cpu().numpy()) * la  # （1，18）
                new_embed.append(generate_data)
                chosen_index.append(minority_point_index)

        idx_neighbor=np.array(idx_neighbor)
        idx_neighbor=np.squeeze(idx_neighbor)
        chosen_index=np.array(chosen_index)

        new_embed =torch.tensor(np.array(new_embed))
        new_embed = torch.squeeze(new_embed)
        new_embed=new_embed.to(embed.device)

        new_labels = labels.new(torch.Size((new_embed.shape[0], 1))).reshape(-1).fill_(label)
        idx_new = idx_train.new(np.arange(embed.shape[0], embed.shape[0] + new_embed.shape[0]))
        idx_new =idx_new.to(embed.device)

        embed = torch.cat((embed, new_embed), 0)
        labels = torch.cat((labels, new_labels), 0)
        idx_train_append=torch.cat((idx_train_append, idx_new), 0)


        if adj is not None:
            if adj_new is None:
                adj_new = adj.new(torch.clamp_(adj[chosen_index, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
            else:
                temp = adj.new(torch.clamp_(adj[chosen_index, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                adj_new = torch.cat((adj_new, temp), 0)

    idx_train = torch.cat((idx_train, idx_train_append), 0)
    idx_train=idx_train.long()

    if adj is not None:
        if adj_new is not None:
            add_num = adj_new.shape[0]
            new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
            new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
            new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
            new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

            return embed, labels, idx_train, new_adj.detach()
        else:
            return embed, labels, idx_train, adj.detach()

    else:
        return embed, labels, idx_train


def cal_knn(data, others, K):

    p = np.sum(np.square(others.detach().cpu().numpy()),axis=1)#（143，）
    q = np.sum(np.square(data.detach().cpu().numpy()), axis=1)#（1，）
    #reshape(1,-1)转化成1行;reshape(-1,1)转换成1列
    distance = -2 * np.dot(data.detach().cpu().numpy(), others.detach().cpu().numpy().T) + p.reshape(1, -1) + q.reshape(-1, 1)#（1，143）
    ner_index = distance.argsort()[:, 1:K + 1]
    return ner_index, distance

def remove_node(class_num_list, prev_out_local, idx_info, idx_train,labels,tau=2):
    class_num_list = torch.tensor(class_num_list)
    sorted_n_data, indices = torch.sort(class_num_list, descending=True)
    n_cls=len(class_num_list)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(class_num_list))[indices][torch.tensor(inv_indices)] - torch.arange(
        len(class_num_list))).sum().abs() < 1e-12

    class_num_list_idx = []
    idx_info_=[]
    for i in range(n_cls):
        data_num=(labels[idx_train]==i).sum().item()
        class_num_list_idx.append(int(data_num))

        cls_indices=set(idx_info[i].tolist())&set(idx_train.tolist())#
        #cls_indices=np.intersect1d(idx_info[i].cpu().numpy(),idx_train_sep.cpu().numpy())
        idx_info_.append(cls_indices)
    class_num_list_idx = np.array(class_num_list_idx)
    class_num_list_idx = class_num_list_idx[inv_indices]
    # 将集合转换为列表
    idx_info_sep = [list(s) for s in idx_info_]
    #idx_info_sep=torch.tensor(list_of_sets, dtype=torch.int64)
    #idx_info_sep=torch.tensor(idx_info_sep)

    max_num, n_cls_idx = max(class_num_list_idx), len(class_num_list_idx)
    max_num = sum(class_num_list_idx) / n_cls_idx
    sampling_list =torch.tensor(class_num_list_idx) - max_num * torch.ones(n_cls_idx )

    prev_out_local = F.softmax(prev_out_local / tau, dim=1)#(422,64)
    prev_out_local = prev_out_local.cpu()

    idx_train_sep_list=idx_train.cpu().tolist()
    local2global = {i: idx_train_sep_list[i] for i in range(len(idx_train_sep_list))}
    global2local = dict([val, key] for key, val in local2global.items())
    idx_info_list = [item for item in idx_info_sep ]
    idx_info_local = [torch.tensor(list(map(global2local.get, k))) for k in idx_info_list]

    idx_train_remove=[]

    #remove_idx_list = [[] for _ in range(n_cls)]
    for cls_idx, num in enumerate(sampling_list):
        num = int(num.item())
        if num <= 0:
            continue

        # first sampling越不重要的样本越容易被采样
        prob = 1 - prev_out_local[idx_info_local[cls_idx]][:, cls_idx].squeeze()
        src_idx_local = torch.multinomial(prob + 1e-12, num,replacement=True)
        src_idx = idx_train[idx_info_local[cls_idx][src_idx_local]]
        #remove_idx_list[cls_idx]=list(src_idx.numpy())

        idx_train_remove.append(src_idx)

    idx_train_remove=torch.cat(idx_train_remove)
    idx_train_remove=torch.unique(idx_train_remove)

    idx_train = torch.tensor(list(set(idx_train.tolist()) - set(idx_train_remove.tolist())))
    #idx_train_new=np.delete(idx_train.cpu().numpy(),idx_train_remove.cpu().numpy())
    #idx_train_new=torch.from_numpy(idx_train_new)
    return idx_train

def adasyn(embed, labels, idx_train,adj=None,beta=1, K=5):

    adj_new=None
    c_largest = labels.max().item()
    idx_train_append = torch.Tensor()
    idx_train_append = idx_train_append.to(embed.device)
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    for label in range(c_largest):
        chosen=idx_train[labels[idx_train] == label]
        G = (avg_number - chosen.shape[0]) * beta
        if G<=0:
            continue
        ratio = []
        idx_neighbor=[]
        chosen_embed = embed[idx_train, :]
        p = np.sum(np.square(chosen_embed.detach().cpu().numpy()), axis=1)
        distance = -2 * np.dot(chosen_embed.detach().cpu().numpy(), chosen_embed.detach().cpu().numpy().T) + p.reshape(
            1, -1) + p.reshape(-1, 1)


        for index,d in enumerate(idx_train):
            if labels[index]==label:
                ner_index = distance[index].argsort()[1:K + 1]
                label_ner = labels[ner_index]
                r = (K-label_ner[label_ner == label].shape[0] )/ K
                ratio.append([index, r,ner_index])
            else:
                continue
        r = [ri[1] for ri in ratio]
        ratio_sum = sum(r)

        if ratio_sum == 0:
            print('data is easy to classify! No necessary to do ADASYN')
            return embed, labels, idx_train, adj_new.detach()

        g = [round(ri[1] / ratio_sum * G) for ri in ratio]
        new_embed=[]
        chosen_index=[]

        for index1, info in enumerate(ratio):
            minority_point_index = info[0]
            x_i = embed[minority_point_index]
            #x_i=x_i.cpu().detach()
            ner = cal_knn(torch.unsqueeze(x_i, 0), embed[labels == label], K)[0]
            for j in range(0, g[index1]):
                random_index = np.random.choice(ner.shape[1])
                la = np.random.ranf(1)
                idx_neighbor.append(ner[:, random_index])
                x_zi = embed[labels == label][ner[:, random_index]]
                #x_zi=torch.squeeze(x_zi)
                generate_data = x_i.detach().cpu().numpy() + (x_zi.detach().cpu().numpy()- x_i.detach().cpu().numpy()) * la  # （1，18）
                new_embed.append(generate_data)
                chosen_index.append(minority_point_index)

        idx_neighbor=np.array(idx_neighbor)
        idx_neighbor=np.squeeze(idx_neighbor)#
        chosen_index=np.array(chosen_index)

        new_embed =torch.tensor(np.array(new_embed))
        new_embed = torch.squeeze(new_embed)
        new_embed=new_embed.to(embed.device)

        new_labels = labels.new(torch.Size((new_embed.shape[0], 1))).reshape(-1).fill_(label)
        idx_new = idx_train.new(np.arange(embed.shape[0], embed.shape[0] + new_embed.shape[0]))
        idx_new =idx_new.to(embed.device)

        embed = torch.cat((embed, new_embed), 0)
        labels = torch.cat((labels, new_labels), 0)
        idx_train_append=torch.cat((idx_train_append, idx_new), 0)


        if adj is not None:
            if adj_new is None:
                adj_new = adj.new(torch.clamp_(adj[chosen_index, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
            else:
                temp = adj.new(torch.clamp_(adj[chosen_index, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                adj_new = torch.cat((adj_new, temp), 0)

    idx_train = torch.cat((idx_train, idx_train_append), 0)
    idx_train=idx_train.long()

    if adj is not None:
        if adj_new is not None:
            add_num = adj_new.shape[0]
            new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
            new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
            new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
            new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

            return embed, labels, idx_train, new_adj.detach()
        else:
            return embed, labels, idx_train, adj.detach()

    else:
        return embed, labels, idx_train

def get_step_split(imb_ratio, valid_each, labeling_ratio, all_idx, all_label, nclass):
    base_valid_each = valid_each

    head_list = [i for i in range(nclass//2)]

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    base_train_each = int( len(all_idx) * labeling_ratio / (t_num + h_num * imb_ratio) )

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list:
        idx2train[i_h] = int(base_train_each * imb_ratio)
        idx2valid[i_h] = int(base_valid_each * 1)

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list:
        idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []

    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:break

    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==total_valid_size:break

    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx, valid_idx, test_idx, train_node

def get_amazon(labels,all_idx,nclass):

    val_idx = []
    test_idx = []
    idx2train,idx2valid,idx2test = {},{},{}
    total_train_size = 0
    total_valid_size = 0
    class_list = [j for j in range(nclass)]

    for i in class_list:
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        c_num = len(c_idx)
        idx2train[i] = int(c_num *0.1)
        idx2valid[i] = int(c_num *0.1)
        idx2test[i]  = int(c_num *0.8)

        total_train_size += idx2train[i]
        total_valid_size += idx2valid[i]

    train_idx = []
    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    label=labels.cpu().detach().numpy()
    for iter1 in all_idx:
        iter_label = label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            #iter_label = train_label[iter]
            train_list[iter_label] += 1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)
        if sum(train_list) == total_train_size: break

    assert sum(train_list) == total_train_size

    after_train_idx = list(set(all_idx) - set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx = []
    for iter2 in after_train_idx:
        iter_label = label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label] += 1
            valid_idx.append(iter2)
        if sum(valid_list) == total_valid_size: break

    test_idx = list(set(after_train_idx) - set(valid_idx))

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(valid_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx,train_node