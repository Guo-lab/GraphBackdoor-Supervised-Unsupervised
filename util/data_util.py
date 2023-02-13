import networkx as nx
import numpy as np
import random
import torch
import pdb
from sklearn.model_selection import StratifiedKFold


class Struc2VecGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        ''' g: a networkx graph                                
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop) 
        '''
        self.g = g
        self.label = label
        self.node_tags = node_tags
        self.node_features = 0
        
        self.edge_mat = 0
        self.neighbors = []
        self.max_neighbor = 0





def load_data(dataset, degree_as_tag):
    ''' dataset: name of dataset; degree_as_tag: Use degree as node tags '''
    print('# -------------------- Loading Data --------------------- #')
    g_list     = []
    label_dict = {}
    feat_dict  = {}
    # ------------------------------------------------------------------- #
    # ---- Load Networkx graph with nodes, edges, and graph labels ------ # 
    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())  # 图的数量
        for i in range(n_g):
            row  = f.readline().strip().split()
            n, l = [int(w) for w in row] # 结点的数量 和 图对应的标签label
            if not l in label_dict:      # label是新的
                mapped = len(label_dict)
                label_dict[l] = mapped
                
            g             = nx.Graph()   # 新建 Graph
            node_tags     = []           # feature 的 int 表示
            node_features = []
            n_edges       = 0
            
            for j in range(n):
                g.add_node(j)            # 把每一个节点加进来
                row = f.readline().strip().split()
                if int(row[1]) + 2 == len(row):  # 无 node attributes 把 degree 作为 feature
                    row, attr = [int(w) for w in row], None
                else:
                    row, attr = [int(w) for w in row[ : int(row[1]) + 2]], np.array([float(w) for w in row[int(row[1]) + 2 : ]])
                #----
                if not row[0] in feat_dict:     # row[0] node tag
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                #---- 
                node_tags.append(feat_dict[row[0]]) # row[0] 对应第几个加入到 dict 中
                if int(row[1]) + 2 > len(row):  # 有 node attributes
                    node_features.append(attr)  # np.array
                #----
                n_edges += row[1]               # row[1] 存了边数量
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])       # [j, k]
            #-------- node 遍历结束
            if node_features != []:
                node_features     = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features     = None
                node_feature_flag = False
            #--------
            assert len(g) == n # 确定节点数信息正常
            if n < 400:        # 400 个节点以下的 g 打包成 Graph 存入list
                g_list.append(Struc2VecGraph(g, l, node_tags))
                
    # -----------------------------
    # -- add neighbors, labels and edge_mat --
    # -----------------------------
    for g in g_list: 
        # <Struc2VecGraph>: 一个个的图g, 图的label, 
        #                   node_tags图节点的标签, node_features, 
        #                   edge_mat, neighbors, max_neighbor
        g.neighbors = [[] for i in range(len(g.g))] # 为 networkx graph 的节点创建空[]
        for i, j in g.g.edges():                    
            g.neighbors[i].append(j)                # i 的邻居
            g.neighbors[j].append(i)                # j 的邻居
        # ---- max_neighbor 找到 degree 最大的
        degree_list = []
        for i in range(len(g.g)):
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)           # 改图节点具有的最大邻居数
        g.label = label_dict[g.label]               # 图的 label
        # ---- edge_mat
        edges = [list(pair)  for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])    # 双向边 
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1) #'[[a, b, c, ...], [b, a, b, ...]]
    
    # ------------------------
    # -- 把 degree 作为 tag --
    # ------------------------
    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values()) # using degree of nodes as tags for each node
    
    # --------------------------------
    # - Extracting unique tag labels -
    # --------------------------------
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))
    tagset     = list(tagset)
    max_degree = max(tagset)
    tag2index  = {i: i for i in range(max_degree + 1)} #'{0: 0, 1: 1, 2: 2, 3: 3, 4: 4} from 0 to max_degree
    
    # --------------------------
    # -- Create node_features --
    # --------------------------
    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tag2index)) # len(nodes) * len(numbers of different degree)
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1 # degree one-hot

    print('# ---------------------- Classes: %d --------------------- #' % len(label_dict))
    print('# --------------- Numbers of Node Tag: %d ---------------- #' % len(tag2index))
    print("# ---------------- Numbers of Graphs: %d --------------- #" % len(g_list))
    return g_list, len(label_dict), tag2index # <Struc2VecGraph>, {}, {} 





def separate_data(graph_list, seed, fold_idx):
    """ graph_list: [<Struc2VecGraph>, ...]
        seed      : int (for randomness)  
        fold_idx  : the index of fold in 10-fold validation. Less than 10.
    """
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    # ---------------------- #
    # -- 使用 skf 划分idx ----#
    # ---------------------- #
    labels   = [graph.label for graph in graph_list]     # []存入每一个graph的 图label
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels): # 循环 3 次 = 3 组 train_index, test_index
        idx_list.append(idx)
    # ----------------------------------------- #
    # -- 一共 n_splits 组 train_idx, test_idx -- #
    # ---- 分配 idx ------ #
    train_idx, test_idx = idx_list[fold_idx]             # 采用第 fold_idx + 1 组
    #print(train_idx, test_idx) => [], []
    # ------------------- #
    # ---- 对应 graph ---- # 
    # ------------------- #
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list  = [graph_list[i] for i in test_idx]
    return train_graph_list, test_graph_list, train_idx, test_idx