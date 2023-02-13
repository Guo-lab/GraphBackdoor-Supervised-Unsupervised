import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rich.progress import track
from models.graphnn import GraphNN
import pickle # 无需安装 pickle 模块，因为它已经与 Python 3.x 一起安装。只需要做 import pickle
#import heartrate; heartrate.trace(browser=True)
from util.config import get_default_args
from util.data_util import *
from util.inject import *
warnings.filterwarnings("ignore")

from dgl.data.tu import TUDataset
def create_graph_classification_dataset(dataset_name):
    name = {
        "COLLAB": "COLLAB",
        "MUTAG" : "MUTAG",
    }[dataset_name]
    dataset = TUDataset(name)
    dataset.num_labels = dataset.num_labels[0]
    dataset.graph_labels = dataset.graph_labels.squeeze()
    return dataset


def main():
    dataset = create_graph_classification_dataset("MUTAG")
    #print(dataset)
    #print(len(dataset)) # 188
    print(dataset[0][0], dataset[0][1], "\n")
    #print(dataset.graph_lists)



if __name__ == '__main__':
    main()
    #skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    
    #labels = [2,3,1,3,2,1,3,3,2,1,3] 
    #x=np.zeros(len(labels))
    #idx_list = []
    #y = 5
    #for idx in skf.split(x, labels):
    #    idx_list.append(idx)
    #print(idx_list)
    
    args = get_default_args().parse_args()
    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    
    graphs, num_classes, tag2index = load_data(args.dataset, args.degree_as_tag)
    clean_train_graphs, clean_test_graphs, train_idx, test_idx = separate_data(graphs, args.seed, args.fold_idx)
    # [], [], [], []
    print('#-- clean_train_graphs: ', len(clean_train_graphs), '    clean_test_graphs: ', len(clean_test_graphs), '--#')
    
    # 干净的测试图 不是目标的图 = 即将污染的图 #!之后不需要?
    backdoor_test_graphs_being_injected = [graph for graph in clean_test_graphs if graph.label != args.target]
    print('#-- test clean graphs: ', len(backdoor_test_graphs_being_injected))
    if args.backdoor:
        train_backdoor, test_backdoor_labels = backdoor_graph_generation_random_supervised(
            args.dataset, 
            args.degree_as_tag,
            args.frac, args.triggersize, 
            args.seed, args.fold_idx, 
            args.target,
            args.graphtype, args.prob, args.K, 
            tag2index)

        for g in test_backdoor_labels:
            g.label = args.target