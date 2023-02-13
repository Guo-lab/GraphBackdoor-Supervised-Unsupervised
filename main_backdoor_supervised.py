import argparse
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rich.progress import track
from models.graphnn import GraphNN
import pickle
import optuna

# import heartrate; heartrate.trace(browser=True)
from util.config import get_default_args
from util.data_util import *
from util.inject import *

warnings.filterwarnings("ignore")

criterion = nn.CrossEntropyLoss()


def train(args, model, device, train_graphs, optimizer, epoch, tag2index): # 包括目标类干净的训练图和其他训练图(一部分已经被注入trigger并relabel) 
    model.train()

    total_iters = args.iters_per_epoch
    pbar = track(range(total_iters), description='epoch: %d' % (epoch)) #tqdm(range(total_iters), unit='batch')
    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size] # total_iters 次随机取前 batch_size 个图
        #//print(selected_idx)
        batch_graph = [train_graphs[idx] for idx in selected_idx] #专有 Struct2Vec 的 []
        output = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        #//print(labels)
        # compute loss
        loss = criterion(output, labels)
        #
        # backprop https://blog.csdn.net/PanYHHH/article/details/107361827
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #
        loss = loss.detach().cpu().numpy()
        loss_accum += loss
    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))
    return average_loss





def pass_data_iteratively(model, graphs, minibatch_size=1):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        #//print(sampled_idx) [0]\n[1]\n[2]\n[...
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    #//print(output[0].shape) #' tensor([[0.8007, 0.0506]]), 
    #//print(len(output)) # = len(graphs) when minibatch_size = 1
    return torch.cat(output, 0)


def test(args, model, device, test_graphs, tag2index):
    model.eval()

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    #//print(labels)
    #//print(labels.view_as(pred))
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item() # labels 转换成和 pred 同大小的 tensor 再找到相等的
    acc_test = correct / float(len(test_graphs))
    print("accuracy test: %f" % acc_test)
    return acc_test





def main():
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
    #//print(backdoor_test_graphs_being_injected)

    print(' -- input dim: ', clean_train_graphs[0].node_features.shape[1]) # column, numbers of diferent degrees

    if args.backdoor:
        train_backdoor, backdoored_test_graphs = backdoor_graph_generation_random_supervised(
            args.dataset, 
            args.degree_as_tag,
            args.frac, args.triggersize, 
            args.seed, args.fold_idx, 
            args.target,
            args.graphtype, args.prob, args.K, 
            tag2index)

        for g in backdoored_test_graphs:
            g.label = args.target

        model = GraphNN(args.num_layers, args.num_mlp_layers, \
                    train_backdoor[0].node_features.shape[1], args.hidden_dim, num_classes, \
                    args.final_dropout, args.learn_eps, \
                    args.graph_pooling_type, args.neighbor_pooling_type, device
                ).to(device)
        model_2=GraphNN(args.num_layers, args.num_mlp_layers, \
                    clean_train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, \
                    args.final_dropout, args.learn_eps, \
                    args.graph_pooling_type, args.neighbor_pooling_type, device
                ).to(device)

        optimizer   = optim.Adam(model.parameters(), lr=args.lr)
        scheduler   = optim.lr_scheduler.StepLR(optimizer,   step_size=50, gamma=0.1)
        optimizer_2 = optim.Adam(model_2.parameters(), lr=args.lr)
        scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, step_size=50, gamma=0.1)
        with open(args.filenamebd, 'w+') as f:
            f.write("Train Acc          |Test Clean Acc     |Test Injected Acc  |Attack Success Rate\n")
            for epoch in range(1, args.epochs + 1):
                scheduler.step()
                scheduler_2.step()
                avg_loss             = train(args, model, device, train_backdoor, optimizer, epoch, tag2index) #包括目标类干净的训练图和其他训练图(一部分已经被注入trigger并relabel) 
                avg_loss_2           = train(args, model_2, device, clean_train_graphs, optimizer_2, epoch, tag2index)
                
                if epoch % 5 == 0:
                    acc_train          = test(args, model, device, train_backdoor,         tag2index)     #包括目标类干净的训练图和其他训练图(一部分已经被注入trigger并relabel) 
                    acc_test_clean_act = test(args, model_2, device, clean_test_graphs,    tag2index)     #!论文中定义 fc 在干净测试图中的acc <-
                    acc_test_clean     = test(args, model, device, clean_test_graphs,      tag2index)     #?论文中定义 fb 在干净测试图中的acc
                    acc_test_backdoor  = test(args, model, device, backdoored_test_graphs, tag2index)     # X 论文中定义 fb 在干净测试图中的acc 
                    #@ 已经都标记成target label, 如果acc test bd 大则证明，predict的结果错得多，即攻击得成功, 确实，是attack success
                    f.write("%f           |%f           |%f           |%f\n" % (acc_train, acc_test_clean_act, acc_test_clean, acc_test_backdoor))
                    f.flush()
                    
        f = open('saved_model/' + str(args.graphtype) + '_' + str(args.dataset) + '_' + str(args.frac) + '_triggersize_' + str(args.triggersize), 'wb')
        pickle.dump(model, f)
        f.close()

        

if __name__ == '__main__':
    main()