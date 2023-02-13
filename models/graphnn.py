import torch
import torch.nn as nn
import torch.nn.functional as F
#import sys
#sys.path.append("models/")
#from mlp import MLP


class MLP(nn.Module): ## MLP with lienar output
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """num_layers: 神经网络 MLP 的不包括输入层的层数        input_dim:  input features 的维度           
           hidden_dim: hidden units at ALL layers 的维度    output_dim: 预测类别数量
        """
        super(MLP, self).__init__()
        # ============ Initiation =============
        self.linear_or_not = True  # 默认线性模型
        self.num_layers    = num_layers
        if   num_layers <  1:
            raise ValueError("Number of MLPlayers MUST be positive!")
        elif num_layers == 1:      # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:                      # Multi-layer model
            self.linear_or_not = False
            self.linears       = torch.nn.ModuleList()
            self.batch_norms   = torch.nn.ModuleList()
            
            # 对于 Linear 各层分为 初始/中间/输出
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            
            # 对于 BatchNorms 共 numlayers - 1 层
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
        # =====================================
        
    def forward(self, x):
        if self.linear_or_not:     # If linear model
            return self.linear(x)
        else:                      # If MLP
            h = x                  # (num_nodes * input_dim Or ...)
            for layer in range(self.num_layers - 1): 
                # [0, 1, ... nums_layers - 2]
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            # 第 num_layers 层
            return self.linears[self.num_layers - 1](h)
        
              
class GraphNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device):
        '''num_layers:      神经网络中包括输入层的所有层数 (除了输入几个MLP)  num_mlp_layers:     多层感知器中不包括输入层的层数
           input_dim:       input features 的维度                       hidden_dim:         hidden units at ALL layers
           output_dim:      预测 输出类别的种数                           final_dropout:      最后线性层的 dropout ratio 
           learn_eps:  If True, learn epsilon to distinguish center nodes from neighboring nodes. 学习区分中心节点和邻居节点
                       If False, aggregate neighbors and center nodes altogether. 
           neighbor_pooling_type: 聚合邻居方式(mean, average, or max)    graph_pooling_type: 在一个图聚合所有邻居的方式(mean, average) average 可带权重
        '''
        super(GraphNN, self).__init__()
        
        # ============ Initiation ===============
        self.final_dropout         = final_dropout
        self.device                = device
        self.graph_pooling_type    = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps             = learn_eps
        self.num_layers            = num_layers
        self.eps                   = nn.Parameter(torch.zeros(self.num_layers - 1))
        #  ------------ List of MLPs ------------
        self.mlps               = torch.nn.ModuleList()
        #  --------- List of batchNorms --------- 应用到 MLP 的输出 也就是 最后预测线性层的输入
        self.batch_norms        = torch.nn.ModuleList()
        #  ----------- Linear function ---------- 将不同层(图聚合时)的隐藏表示映射到预测分数上
        self.linears_prediction = torch.nn.ModuleList()
        #  - Each layer of MLPs and BNs, Linear -
        for layer in range(self.num_layers):
            if   layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim,  hidden_dim, hidden_dim))
                self.linears_prediction.append(nn.Linear(input_dim,  output_dim)) # 对齐，使得分类分数可加减 
            elif layer == self.num_layers - 1:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim)) # 对齐 
                break
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))                
        # ========================================
        # 


    def next_layer(self, h, layer, Adj_block = None, isEps = False):
        #@pooling neighboring nodes and center nodes altogether / separately by epsilon reweighting.   
        # --- If sum or average pooling ---
        #@ Adj_block (num_nodes * num_nodes)
        pooled = torch.spmm(Adj_block, h) 
        if self.neighbor_pooling_type == "average":                                             # degree
            degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device)) # [_|_] * [_]
            pooled = pooled/degree                                                              # [_|_]   [ ]
        # ---------------------------------
        
        # representation of neighboring and center nodes
        if isEps == True:
            # Reweights the center node representation 
            # when aggregating it with its neighbors ?
            pooled = pooled + (1 + self.eps[layer]) * h 

        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        return F.relu(h)


    def __preprocess_neighbors_sumavepool(self, batch_graph):
        #@create block diagonal sparse matrix
        # ----------- 创建每个图邻接矩阵的 block 位置 -----------
        edge_mat_list = []                                  # 边在所有图中的
        start_idx     = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))   # 与 graph pooling 的 idx 类似
            #'append tensor([[320, 320, 321, ..., 329, 328, 334], []])
            edge_mat_list.append(graph.edge_mat + start_idx[i]) 
        #//print(edge_mat_list)
        Adj_block_idx  = torch.cat(edge_mat_list, 1)
        #//print(Adj_block_idx)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1]) #宽: 边的数量, 有边则为1
        # ----------------------------------------------------
        
        # --------- 不学中心节点的权重, 加自边 ------------
        # Add self-loops in the adjacency matr if learn_eps == False
        # i.e., aggregate center nodes and neighbor nodes altogether
        if not self.learn_eps:               # if learn_eps == False
            num_node       = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)]) #'[[0,1,...],[0,1,...]]
            elem           = torch.ones(num_node)
            Adj_block_idx  = torch.cat([Adj_block_idx, self_loop_edge], 1)        # 2 * (num_nodes * 2)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)                 # (2 * num_nodes) * 1  
            #//print(Adj_block_idx.shape, Adj_block_elem.shape)
            #//print(self_loop_edge)
        # ----------------------------------------------
        
        #@return Size 存储几个图拼在一起的邻接矩阵
        return torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]])).to(self.device)
    
    
    def __preprocess_graphpool(self, batch_graph):
        #@create sum/average pooling sparse matrix over entire nodes in each graph (num_graphs x num_nodes)
        # -------- start_idx 每个图第一个节点在所有图中的编号 ---------
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            
        # --------- 这些图节点的pooling乘积项 ----------
        idx  = [] # sparse 索引 第几个图的多少号节点
        elem = [] # sparse 非 0 的值
        for i, graph in enumerate(batch_graph): 
            if self.graph_pooling_type == "average": 
                elem.extend([1./len(graph.g)] * len(graph.g))
            else: # sum pooling extend [] * 节点的数量
                elem.extend([1] * len(graph.g)) 
            #'idx extend [0, 0], [0, 1], ..., [0, 24], [1, 25], ...
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)]) 
            
        # ----------- sparse matrix 构建 -------------
        elem       = torch.FloatTensor(elem)
        idx        = torch.LongTensor(idx).transpose(0,1) 
        # @return sparse(idx, elem, Size: 原 matrix 大小)
        return torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]])).to(self.device)
    
    
    def forward(self, batch_graph):
        X_concat   = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device) # 将所有 batch_size 个 graph 的 feature 拼接
        # print(X_concat.shape) #@ num_nodes * feature_dim (max_degree + 1)
        
        # ------ graph pooling 的 乘积 sparse 矩阵 ------
        graph_pool = self.__preprocess_graphpool(batch_graph)
        # ---------------------------------------------
        
        # ---- neighbor pooling 的 乘积 sparse 矩阵 -----
        if not self.neighbor_pooling_type == "max":
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)
        # ---------------------------------------------
        
        # ------- 每层的隐藏表示 （记录 tmp）-------- #@list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h          =  X_concat
        # ---- 暂时不考虑max ----
        # Adj_block 是对节点 pool
        for layer in range(self.num_layers - 1): 
            if   not self.neighbor_pooling_type == "max" and     self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block, isEps = True)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)
            hidden_rep.append(h)
        # ---------------------------------------
        
        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):   #  perform pooling over all nodes in each graph in every layer
            pooled_h = torch.spmm(graph_pool, h) #  (系数 * data_X)
            #print(layer)                        #@ layer 0 => X_concat
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)
            #//print(score_over_layer.shape) # 32 * 2
        return score_over_layer