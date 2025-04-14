
# point to voxel distance network

import sys
import os
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable



###############################################################################################
#                                           PointNet structure
###############################################################################################


# STN: spacial transformer network
# STN3d: STN for 3D point cloud.
# STNkd: STN for k-dim features

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.float() if x.dtype != torch.float32 else x    # convert to float32.
        batchsize = x.size()[0]
        
        # TODO: Dongyan Added. Why?
        x = x.transpose(2,1)
        # Dongyan Added

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.float() if x.dtype != torch.float32 else x    # convert to float32.
        assert x.dim()==3, '[Error] Input must be of shape (B, D, N), you are missing `batch`'
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        
        # TODO: Dongyan removed. Why?
        # x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)        # bmm: batch matrix multiply, 
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat





###############################################################################################
#                                           My Own P2V Net
###############################################################################################



# position encoder: input x, output its encoding result
class PositionalEncoder():
    def __init__(self, level:int):
        self.n_freqs = level
        ## remove lambda function. Convertion may have bugs.
        # self.freq_bands = torch.linspace(2.**0., 2.**(level - 1), self.n_freqs)
        # self.embed_fns = [lambda x : x]
        # # Alternate sin and cos
        # for freq in self.freq_bands:
        #     self.embed_fns.append(lambda x, freq=freq: x*torch.sin(x * freq))       # TODO: x*torch
        #     self.embed_fns.append(lambda x, freq=freq: x*torch.cos(x * freq))
            
    def encode(self, xyz):
        # # 对于每个坐标轴 x, y, z 进行批量处理
        # x = torch.cat([fn(xyz[:, 0:1]) for fn in self.embed_fns], dim=-1)  # 仍然传递 (batch_size, 1)，输出是 (batch_size, n)
        # y = torch.cat([fn(xyz[:, 1:2]) for fn in self.embed_fns], dim=-1)  # 仍然传递 (batch_size, 1)，输出是 (batch_size, n)
        # z = torch.cat([fn(xyz[:, 2:3]) for fn in self.embed_fns], dim=-1)  # 仍然传递 (batch_size, 1)，输出是 (batch_size, n)
        # m = torch.cat([x, y, z], dim=-1)
        # return m
        # Now just use original coor.
        return xyz




class P2VDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        global_feat_dim = 1024
        position_feat_dim = 3
        self.feat_dim = global_feat_dim + position_feat_dim            # 1027 input dim
        self.fc1 = nn.Linear(self.feat_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)                                    # output dim: 1*3 p2v + weight
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, feature : torch.tensor):
        x = F.relu(self.fc1(feature))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # seperate p2v and weight function.
        p2v = x[:,:3]
        weight = x[:, 3]
        weight = torch.sigmoid(weight)  # normalize the weight. TODO: check if this is necessary? 
        return p2v, weight



class Point2VoxelNet(nn.Module):
    def __init__(self, voxel_size):
        super().__init__()
        self.feat = PointNetEncoder()                       # use the same name as PointNet, for easier paramter loading.
        self.position_encoder = PositionalEncoder(level=1)       # Not implemented yet.
        self.decoder = P2VDecoder()
        self.voxel_size = voxel_size            # used to check input

    def forward(self, points:torch.tensor, p:torch.tensor):
        
        points = points.float() if points.dtype != torch.float32 else points    # convert to float32.
        p = p.float() if p.dtype != torch.float32 else p    # convert to float32.
        # check the input data are normalized to identity voxel
        assert max(abs(p)[0]) <= self.voxel_size, "The input query-point is not normalized to the identity voxel."
        assert abs(p).max() <= self.voxel_size, "The input point cloud is not normalized to the identity voxel."

        global_feature, _, _ = self.feat(points)
        position_encoding = self.position_encoder.encode(p)
        x = torch.cat([global_feature, position_encoding], dim=1)
        p2v, weight = self.decoder(x)
        return p2v, weight, global_feature




###############################################################################################
#                                           My Loss function
###############################################################################################


# 异方差回归损失函数（Heteroscedastic Regression Loss）
class MyLossFunction(nn.Module):
    def __init__(self, use_weight, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.use_weight = use_weight
        self.alpha = 0.1

        self.mse_loss = nn.MSELoss()


    def forward(self, gt_p2v:torch.Tensor, pred_p2v:torch.Tensor, 
                    gt_weight:torch.Tensor,  pred_weight:torch.Tensor) -> torch.Tensor:

        assert gt_p2v.shape == pred_p2v.shape, "different predict and gt p2v size"
        assert len(pred_weight.shape) == 1, "error weight shape"
        batch_size = gt_p2v.size(0)

        

        # ## Loss 1. p2v using NLL, and weight using KL
        # pred_weight = pred_weight.view(-1, 1)
        # gt_weight = gt_weight.view(-1, 1)
        # pred_var = 1 - pred_weight
        # gt_var = 1 - gt_weight
        
        # L_p2v_part1 = torch.sum((pred_p2v - gt_p2v)**2/(pred_var**2 + self.epsilon), dim=-1).mean()
        # L_p2v_part2 = torch.log(pred_var**2 + self.epsilon).mean()
        # L_p2v = 0.5*(L_p2v_part1 + L_p2v_part2)

        # L_weight = (pred_var - gt_var)**2 / gt_var**2 + torch.log(gt_var**2+self.epsilon)
        # L_weight = L_weight.mean()

        # loss1 = L_p2v + self.alpha * L_weight


        # ## Loss 2. p2v use MSE
        # loss2 = self.mse_loss(pred_p2v, gt_p2v)

        ## Loss 3. p2v add gt_weight, and push pred_weight -> gt_weight
        """
        pred_p2v: 预测的3D向量, shape [batch_size, 3]
        pred_weight: 预测的可信度, shape [batch_size]
        gt_p2v: 真实的3D向量, shape [batch_size, 3]
        gt_weight: 真实的可信度, shape [batch_size]
        """
        p2v_diff = (pred_p2v - gt_p2v)**2
        weighted_p2v_diff = p2v_diff * gt_weight.view(-1, 1)       # for each p2v diff, add "weight"
        loss_p2v = torch.sum(weighted_p2v_diff, dim=1)

        loss_weight = (pred_weight - gt_weight)**2

        loss3 = loss_p2v + self.alpha * loss_weight
        loss3 = torch.mean(loss3)

        # output other loss
        debug_loss_weight = torch.mean(torch.abs(pred_weight - gt_weight))
        debug_loss_p2v = torch.mean(torch.norm(pred_p2v-gt_p2v, dim=1))
        debug_loss_p2v_weighted = torch.mean(torch.norm(pred_p2v-gt_p2v, dim=1)*gt_weight)

        return loss3, debug_loss_p2v, debug_loss_p2v_weighted, debug_loss_weight




# Test Loss function.
if __name__ == "__main__":
    B = 4  # 批大小
    use_weight = False
    loss_fn = MyLossFunction(use_weight)
    
    # 模拟输入数据
    gt_p2v = torch.randn(B, 3)          # 真实距离（1x3向量）
    pred_p2v = gt_p2v + 0.1 * torch.randn(B, 3)  # 预测距离（加入噪声）
    pred_weight = torch.sigmoid(torch.randn(B))  # 预测权重（sigmoid限制到0~1附近）
    gt_weight = torch.ones(B)            # 可选的真实样本权重
    
    loss = loss_fn(gt_p2v, pred_p2v, gt_weight, pred_weight)
    print(f"计算得到的异方差损失: {loss.item():.4f}")
