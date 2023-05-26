import math

import torch
from torch import nn


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer("params", torch.tensor([-1]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory", torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )
        print("using queue shape: ({},{})".format(self.queueSize, inputSize))
        # 16384, 64




    def forward(self, q, k):    # (feat_q, feat_k)
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()
        # print(self.params) # tensor([-1])
        

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1)) # debug # view embedding
        # print(l_pos, "in l_pos", l_pos.shape)
        # exit(0)
        l_pos = l_pos.view(batchSize, 1)
        # bmm [ ( 32, 1, 64 ) * (32, 64, 1) ] => (32, 1)
        
        # print(self.memory)
        # neg logit
        queue = self.memory.clone()
        # print(queue.detach(), "\n", queue.detach().shape, q.transpose(1,0).shape)
        # l_neg: mm (16384 * 64, 64 * 32) 
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0)) # https://blog.csdn.net/irober/article/details/113686080
        l_neg = l_neg.transpose(0, 1)
        # print(l_neg.shape) # 32 * 16384

        out = torch.cat((l_pos, l_neg), dim=1)
        # 终于明白这个cat了，字典拼接，l_pos (32*1) + l_neg (32*16384 other keys) 

        if self.use_softmax: # True
            out = torch.div(out, self.T)
            # print("out: ", out, out.shape) #  torch.Size([32, 16385])
            out = out.squeeze().contiguous()
            # print("out: ", out, out.shape) #  torch.Size([32, 16385])
            
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0: 
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous() # https://blog.csdn.net/weixin_43593330/article/details/108405998

        # # update memory
        with torch.no_grad():
            # out_ids = torch.arange(batchSize).cuda()
            out_ids = torch.arange(batchSize)## .cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out

# 2023.4.29
# 1. 交替 loss，epoch 一A一B 
# 2. 超参
# 3. 攻击范围 2%

# 4. 捋思路
# 5. 有目标+训练下游任务
 