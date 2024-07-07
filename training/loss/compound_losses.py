import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
class Regre_loss(nn.Module):
    def __init__(self, w1, w2, w3):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Regre_loss, self).__init__()

        self.weight_w1 = w1
        self.weight_w2 = w2
        self.weight_w3 = w3

        self.loss1 = nn.MSELoss()
        self.loss2 = nn.L1Loss()
        self.loss3 = nn.functional.huber_loss


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        # if (target == 2.0).any():
        #     result = 0
        #     for now_net_output, now_target in zip(net_output, target):
        #         now_index = now_target != 2.0
        #         now_net_output = now_net_output[now_index].unsqueeze(0)
        #         now_target = now_target[now_index].unsqueeze(0)
                
        #         now_loss1 = self.loss1(now_net_output, now_target)
        #         now_loss2 = self.loss2(now_net_output, now_target)
        #         now_loss3 = self.loss3(now_net_output, now_target, delta=1.0)
                
        #         now_result = self.weight_w1 * now_loss1 + self.weight_w2 * now_loss2 + self.weight_w3 * now_loss3
        #         result += now_result

        # else:
        loss1 = self.loss1(net_output, target)
        loss2 = self.loss2(net_output, target)
        loss3 = self.loss3(net_output, target, delta=1.0)
        result = self.weight_w1 * loss1 + self.weight_w2 * loss2 + self.weight_w3 * loss3
        
        # 单独处理 0 1
        # mask = (target == 0) | (target == 1)
        # selected_predictions = net_output[mask]
        # selected_targets = target[mask]
        # loss1_01 = self.loss1(selected_predictions, selected_targets)
        # loss2_01 = self.loss2(selected_predictions, selected_targets)
        # loss3_01 = self.loss3(selected_predictions, selected_targets, delta=1.0)
        # result_01 = self.weight_w1 * loss1_01 + self.weight_w2 * loss2_01 + self.weight_w3 * loss3_01

        # result = result + 0.3 * result_01

        # 根据这些坐标从 A 和 B 中提取相应的值
        # 单独处理 0 1
        # result01 = 0
        # for now_net_output, now_target in zip(net_output, target):
        #     now_index = torch.where((now_target == 0) | (now_target == 1))
        #     now_net_output = now_net_output[now_index[0],].unsqueeze(0)
        #     now_target = now_target[now_index[0],].unsqueeze(0)
        #     now_loss1 = self.loss1(now_net_output, now_target)
        #     now_loss2 = self.loss2(now_net_output, now_target)
        #     now_loss3 = self.loss3(now_net_output, now_target, delta=1.0)
        #     now_result = self.weight_w1 * now_loss1 + self.weight_w2 * now_loss2 + self.weight_w3 * now_loss3
        #     result01 = result01 + now_result
        # w01 = 0.1
        # result = result + w01 * result01
        # print("result:  ",result)
        return result

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.99, gamma=2.0, reduction='mean'):
        """
        初始化Focal Loss。
        参数:
        - alpha (float): 正样本的权重。
        - gamma (float): 调节易分类样本权重的focusing参数。
        - reduction (str): 损失的缩减方式，'none', 'mean' 和 'sum' 中选择。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算Focal Loss。
        参数:
        - inputs (tensor): 模型输出，shape为(N, C)，其中C是类别数。
        - targets (tensor): 真实标签，shape为(N,)，每个值为[0, C-1]之间的整数。
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt是模型对正确类别的预测概率
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class Classification_loss(nn.Module):
    def __init__(self, w1, w2, w3):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Classification_loss, self).__init__()

        self.weight_w1 = w1
        # self.weight_w1 = 0.5
        self.weight_w2 = w2
        self.weight_w3 = w3

        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = FocalLoss()


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        classflag = 1

        if classflag == 4:
            # *********************************************** 分四类   *************************************
            # target_modified = (target == 0) | (target == 1)
            # 修正差分Tensor的形状问题
            # 初始化差分Tensor的填充版本，使其形状与原始Tensor相匹配
            diff_tensor_torch_padded = torch.zeros_like(target)
            diff_tensor_torch = torch.diff(target, dim=1)
            diff_tensor_torch_padded[:, 1:, :] = diff_tensor_torch  # 从第二个元素开始填充差分结果
            diff_tensor_torch_padded[:, 0:1, :] = target[:, 1:2, :] - target[:, 0:1, :]  # 第一个元素的差分处理
            diff_tensor_torch_padded[:, -1:, :] = target[:, -1:, :] - target[:, -2:-1, :]  # 最后一个元素的差分处理

            # 重新初始化new_tensor_torch为2，再次处理增加和减少的趋势
            target_modified = torch.zeros_like(target)
            target_modified[:] = 2  # 假设所有非0和非1的值都是在增加趋势中
            target_modified[diff_tensor_torch_padded > 0] = 1  # 增加趋势
            target_modified[diff_tensor_torch_padded < 0] = 3  # 减少趋势

            # 现在处理值为0和1的情况
            target_modified[target == 0] = 0
            target_modified[target == 1] = 2
            target_modified = target_modified.float().squeeze(-1)
            target_modified = target_modified.view(-1).long()
            net_output = net_output.view(-1, net_output.shape[-1])
            loss1 = self.loss1(net_output, target_modified)
            # target_modified_onehot = F.one_hot(target_modified, num_classes=2).float()
            target_modified_onehot = F.one_hot(target_modified, num_classes=4).float()
            loss2 = self.loss2(net_output, target_modified_onehot)
            result = self.weight_w1 * loss1 + self.weight_w2 * loss2
            # *********************************************** 分四类   *************************************
        elif classflag == 2:
            # *********************************************** 分2类   *************************************
            # target_modified = (target == 0) | (target == 1)
            # 修正差分Tensor的形状问题
            # 初始化差分Tensor的填充版本，使其形状与原始Tensor相匹配
            diff_tensor_torch_padded = torch.zeros_like(target)
            diff_tensor_torch = torch.diff(target, dim=1)
            diff_tensor_torch_padded[:, 1:, :] = diff_tensor_torch  # 从第二个元素开始填充差分结果
            diff_tensor_torch_padded[:, 0:1, :] = target[:, 1:2, :] - target[:, 0:1, :]  # 第一个元素的差分处理
            diff_tensor_torch_padded[:, -1:, :] = target[:, -1:, :] - target[:, -2:-1, :]  # 最后一个元素的差分处理

            # 重新初始化new_tensor_torch为2，再次处理增加和减少的趋势
            target_modified = torch.zeros_like(target)
            target_modified[:] = 2  # 假设所有非0和非1的值都是在增加趋势中
            target_modified[diff_tensor_torch_padded > 0] = 0  # 增加趋势
            target_modified[diff_tensor_torch_padded < 0] = 1  # 减少趋势

            # 现在处理值为0和1的情况
            target_modified[target == 0] = 0
            target_modified[target == 1] = 1
            target_modified = target_modified.float().squeeze(-1)
            target_modified = target_modified.view(-1).long()
            net_output = net_output.view(-1, net_output.shape[-1])
            loss1 = self.loss1(net_output, target_modified)
            target_modified_onehot = F.one_hot(target_modified, num_classes=2).float()
            loss2 = self.loss2(net_output, target_modified_onehot)
            result = self.weight_w1 * loss1 + self.weight_w2 * loss2
            # *********************************************** 分2类   *************************************
        elif classflag == 1:
            target_modified = (target == 0) | (target == 1)
            target_modified = target_modified.float().squeeze(-1)
            target_modified = target_modified.view(-1).long()
            net_output = net_output.view(-1, net_output.shape[-1])
            loss1 = self.loss1(net_output, target_modified)
            target_modified_onehot = F.one_hot(target_modified, num_classes=2).float()
            loss2 = self.loss2(net_output, target_modified_onehot)
            result = self.weight_w1 * loss1 + self.weight_w2 * loss2
        return result
    

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
