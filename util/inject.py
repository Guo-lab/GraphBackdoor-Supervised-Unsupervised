import networkx as nx
import numpy as np
import random
import torch
import pdb
from util.data_util import load_data, separate_data


def backdoor_graph_generation_random_supervised(dataset, degree_as_tag, frac, num_backdoor_nodes, seed, fold_idx, target_label, graph_type, prob, K, tag2index):
    """ dataset:    name of the dataset        degree_as_tag:      use degree as node_tags / node_features   
        frac   :    poison fraction            num_backdoor_nodes: trigger size
        seed, fold_idx: same as global
        target_label: trigger makes clssifier misclassify other labels into target label
        graph_type: graph generation method    prob:               probability for edge creation/rewiring in a specific method
        tag2index:  {}, from 0 to max_degree   #'{0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    """
    if graph_type == 'ER':
        # -- 检查修改边的概率是否符合条件? -- #
        print(np.log(num_backdoor_nodes) / num_backdoor_nodes) # log(x) / x
        assert prob > np.log(num_backdoor_nodes) / num_backdoor_nodes
        # --- random graph generator --- # https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html
        G_gen = nx.erdos_renyi_graph(num_backdoor_nodes, prob)
        nx.write_edgelist(G_gen, 'tmp_data/subgraph_gen/ER_' + str(dataset) + '_triggersize_' + str(num_backdoor_nodes) + '_prob_' + str(prob) + '.edgelist')
        # --创建文件记录 测试图, 训练图 和 训练图节点
        test_graph_file      = open('tmp_data/test_graphs/' + 
            str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        train_graph_file     = open('tmp_data/backdoor_graphs/' +
            str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        train_graph_nodefile = open('tmp_data/backdoor_graphs/' +
            str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphnodes', 'w')
    # -- graph generator generates a new graph -- #
    #print(G_gen.nodes) #'[0, 1, 2, 3]
    #print(G_gen.edges) #'[(0, 1), (1, 2), (1, 3)]
    
    # ------------------ #
    # -- 重新加载 data -- #
    graphs, num_classes, tag2index = load_data(dataset, degree_as_tag)
    # 一开始干净的所有 训练图 和 测试图
    train_graphs, test_graphs, train_idx, test_idx = separate_data(graphs, seed, fold_idx)
    #print(train_idx, test_idx) #@ seed一致, 同一个 fold_idx 对应的划分相同
    print('# ------------ Numbers of Train Graphs: ', len(train_graphs), '------------ #')
    # -------------------#

    # -- 被注入后门的训练图 -- #
    num_backdoor_train_graphs = int(frac * len(train_graphs)) 
    print('[Train.Numbers of Backdoor Graphs (%d %%): ' % (frac * 100), num_backdoor_train_graphs, "]")


    #---------------------------------#
    # -- 在攻击前，分清楚后门注入的标签 -- #
    # any labels -> target_label (backdoor injected)
    train_graphs_with_target_label_indexes        = []
    train_graphs_without_target_label_indexes     = [] 
    # from 0 to num of all train_graph
    for graph_idx in range(len(train_graphs)): 
        if train_graphs[graph_idx].label == target_label:
            train_graphs_with_target_label_indexes.append(graph_idx)
        else:
            train_graphs_without_target_label_indexes.append(graph_idx)
    print(' +-- numbers of clean train graphs with target label:', len(train_graphs_with_target_label_indexes), '\n',
          '+-- numbers of clean train graphs with other labels:', len(train_graphs_without_target_label_indexes))
    # -- 从可以注入攻击的图中 无放回抽样选出 num_backdoor_train_graphs 个图攻击 --#
    rand_backdoor_graph_idx = random.sample(
        train_graphs_without_target_label_indexes,
        k = num_backdoor_train_graphs)
    # -- 被攻击的图编号存储 --#
    train_graph_file.write(" ".join(str(idx) for idx in rand_backdoor_graph_idx))
    train_graph_file.close()
    #---------------------------------#


    # ============================================ #
    # -- 对训练图注入生成的 trigger -- #
    for idx in rand_backdoor_graph_idx:
        #//print(train_graphs[idx].edge_mat)
        # --------------------------------- #
        # -- 该图的最大节点个数 -- #
        num_nodes = torch.max(train_graphs[idx].edge_mat).numpy() + 1 
        if num_backdoor_nodes >= num_nodes:
            # == trigger size 过大 == # 可以在该图中随机取较多的(可重复的)点 因为生成了trigger size这么多的点
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes)
        else:
            # == trigger size 少于图中的点就不能重复取了 == #
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes, replace=False)
        # -- 为每一个随机选出来攻击的图分配选出的结点 一个图存一行 -- #
        train_graph_nodefile.write(" ".join(str(idx) for idx in rand_select_nodes))
        train_graph_nodefile.write("\n")
        # --------------------------------- #
        
        #+-- 注入攻击图所有的原始边 --+
        edges = train_graphs[idx].edge_mat.transpose(1, 0).numpy().tolist() #'[[0, 1], [1, 2], ...]
        #//print(len(edges))
        
        # ----------------------------------------------------- #
        #+-- Remove existing edges 把 trigger 在图中的已有边去掉 --+ 
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])
                if (i, j) in train_graphs[idx].g.edges():
                    train_graphs[idx].g.remove_edge(i, j)  #@ 在原训练干净图中修改
        #//print('after remove:', len(edges))
        # ----------------------------------------------------- #
        
        # ----------------------------------------------------- #
        # map node index [0,1,.., num_backdoor_node-1] to 
        # corresponding nodes in rand_select_nodes 
        # and attach the subgraph 
        # ------- 把随机选取的trigger中的点对应从0开始的index ------ #
        for e in G_gen.edges: # a networkx Graph from nx.erdos_renyi_graph(num_backdoor_nodes, prob)
            #~         the indexes in e begin with 0         ~#
            #~ the indexes in rand are the same as raw graph ~#
            edges.append([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
            edges.append([rand_select_nodes[e[1]], rand_select_nodes[e[0]]])
            train_graphs[idx].g.add_edge(e[0], e[1])       #@ 在原训练干净图中修改
        # --- 随机生成的 ER 图 已在 clean train dataset 中注入 --- #
        # - 使得 idx 对应需要注入攻击的图嵌入 trigger, 之后relabel - #

        # -------------------------
        # ---- 更新受攻击图信息 -----
        # -------------------------
        train_graphs[idx].edge_mat      = torch.LongTensor(np.asarray(edges).transpose())  # 存入 edge_mat
        #//print(train_graphs[idx].edge_mat)
        train_graphs[idx].label         = target_label                                     # relabel
        #//print("before,", train_graphs[idx].node_tags)
        train_graphs[idx].node_tags     = list(dict(train_graphs[idx].g.degree).values())  # 结合 degree, 修正 tag 以及 feature
        #//print("after, ",  train_graphs[idx].node_tags, "\n")
        # 原先根据所有图的 degree 生成的 tag2index
        # 假设生成图不会超过原先所有图的最大 degree?
        train_graphs[idx].node_features = torch.zeros(
            len(train_graphs[idx].node_tags),
            len(tag2index)
        ) 
        #//print(tag2index); print(train_graphs[idx].node_tags)
        for i in range(len(train_graphs[idx].node_tags)):
            if train_graphs[idx].node_tags[i] > len(tag2index) - 1:
                train_graphs[idx].node_tags[i] =  len(tag2index) - 1
        train_graphs[idx].node_features[ \
            range(len(train_graphs[idx].node_tags)), \
            [tag2index[tag] for tag in train_graphs[idx].node_tags] \
        ] = 1
    
    train_graph_nodefile.close()
    # ------- 对 idx 遍历结束 ------- #
    # ============================================ #

    #+-- 可以观察到 relabel 的情况 --+#
    #//train_labels = torch.LongTensor([graph.label for graph in train_graphs]) #'tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1])


    # ------------------------------------------------ #
    # ---------- 与训练图注入 trigger 过程类似 ---------- #
    # 但是仅保存下了注入 trigger 的图编号信息, 省下了节点信息 #
    test_graphs_with_target_label_indexes    = []
    test_graphs_without_target_label_indexes = []
    for graph_idx in range(len(test_graphs)):
        if test_graphs[graph_idx].label != target_label: # 不是目标类即需要注入trigger构成所谓测试集  
            test_graphs_without_target_label_indexes.append(graph_idx)
        else:
            test_graphs_with_target_label_indexes.append(graph_idx)
    print(' +-- numbers of test graphs with target label:', len(test_graphs_with_target_label_indexes), '\n'
          ' +-- numbers of test graphs with other labels:', len(test_graphs_without_target_label_indexes))
    test_graph_file.write(" ".join(str(idx) for idx in test_idx))
    test_graph_file.close()
    # backdoored testing dataset, consists of test graphs.
    # ----------------------------------------------- # 
    # -- 对测试图中所有不是目标类的图注入 trigger -- #
    for idx in test_graphs_without_target_label_indexes:
        num_nodes = torch.max(test_graphs[idx].edge_mat).numpy() + 1
        if num_backdoor_nodes >= num_nodes:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes)
        else:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes, replace=False)
        edges = test_graphs[idx].edge_mat.transpose(1, 0).numpy().tolist() #'[[0, 1], [0, 5], [1, 2], ...]

        #---- 去除 trigger 在图中的原有边 ----#
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])
                if (i,j) in test_graphs[idx].g.edges():
                    test_graphs[idx].g.remove_edge(i, j)
        #---- 把 generator 生成的 trigger 的边放在图中对应的位置 ----#
        # ------- generaotr 的边的关系可以和训练图共用 ---------#
        for e in G_gen.edges:
            edges.append([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
            edges.append([rand_select_nodes[e[1]], rand_select_nodes[e[0]]])
            test_graphs[idx].g.add_edge(e[0], e[1])
        # -------------------------
        # ---- 更新受攻击图信息 -----
        # -------------------------
        test_graphs[idx].edge_mat      = torch.LongTensor(np.asarray(edges).transpose())
        test_graphs[idx].node_tags     = list(dict(test_graphs[idx].g.degree).values())
        test_graphs[idx].node_features = torch.zeros(len(test_graphs[idx].node_tags), len(tag2index))
        for i in range(len(test_graphs[idx].node_tags)):
            if test_graphs[idx].node_tags[i] > len(tag2index) - 1:
                test_graphs[idx].node_tags[i] =  len(tag2index) - 1        
        test_graphs[idx].node_features[ \
            range(len(test_graphs[idx].node_tags)), \
            [tag2index[tag] for tag in test_graphs[idx].node_tags] \
        ] = 1

    backdoored_test_graphs = [graph for graph in test_graphs if graph.label != target_label]
    """ #返回参数 #@train_graphs:  包括目标类干净的训练图和其他训练图(一部分已经被注入trigger并relabel) 
        @backdoored_test_graphs:  所有注入trigger的非目标类
    """
    return train_graphs, backdoored_test_graphs