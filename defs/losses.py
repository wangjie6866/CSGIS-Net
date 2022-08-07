import torch
import defs.Myloss as Myloss 
from torch import nn
from torch.nn import L1Loss, MSELoss
from torch.nn.functional import max_pool2d
from .pytorch_ssim import SSIM
from torchvision.models.vgg import vgg16

__all__ = [
    "edgeLoss",
    "mL1Loss",
    "mSSIMLoss",
    "TVLoss",
    "GDLoss",
    "const_loss",
    "const_loss2",
    "ContrastLoss"
]
class edgeLoss(nn.Module):
    def __init__(self):
        super(edgeLoss, self).__init__()
    def forward(self, prediction, label):
        label = label.long()
        mask = (label != 0).float()
        num_positive = torch.sum(mask).float()
        num_negative = mask.numel() - num_positive
        # print (num_positive, num_negative)
        mask[mask != 0] = num_negative / (num_positive + num_negative)
        mask[mask == 0] = num_positive / (num_positive + num_negative)
        cost = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(), label.float(), weight=mask, reduce=False)
        return torch.sum(cost)

class mL1Loss(nn.Module):
    def __init__(self):
        super(mL1Loss, self).__init__()
        self.L_0 = nn.L1Loss()
        self.L_1 = nn.L1Loss()
        self.L_2 = nn.L1Loss()
        self.L_3 = nn.L1Loss()
    def forward(self, prediction, label):
        p1, l1 = max_pool2d(input=prediction, kernel_size=2, stride=2), max_pool2d(input=label, kernel_size=2, stride=2)
        p2, l2 = max_pool2d(input=p1, kernel_size=2, stride=2), max_pool2d(input=l1, kernel_size=2, stride=2)
        p3, l3 = max_pool2d(input=p2, kernel_size=2, stride=2), max_pool2d(input=l2, kernel_size=2, stride=2)
        return self.L_0(prediction, label) + self.L_1(p1, l1) + self.L_2(p2, l2) + self.L_3(p3, l3)

class mSSIMLoss(nn.Module):
    def __init__(self):
        super(mSSIMLoss, self).__init__()
        self.L_0 = SSIM()
        self.L_1 = SSIM()
        self.L_2 = SSIM()
        self.L_3 = SSIM()
    def forward(self, prediction, label):
        p1, l1 = max_pool2d(input=prediction, kernel_size=2, stride=2), max_pool2d(input=label, kernel_size=2, stride=2)
        p2, l2 = max_pool2d(input=p1, kernel_size=2, stride=2), max_pool2d(input=l1, kernel_size=2, stride=2)
        p3, l3 = max_pool2d(input=p2, kernel_size=2, stride=2), max_pool2d(input=l2, kernel_size=2, stride=2)
        return self.L_0(prediction, label) + self.L_1(p1, l1) + self.L_2(p2, l2) + self.L_3(p3, l3)

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class GDLoss(nn.Module):
    def __init__(self):
        super(GDLoss, self).__init__()

    def forward(self,x,mask):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        w_g = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        w_e = x[:, :, :, 0] - x[:, :, :, w_x-1]
        w_e = torch.unsqueeze(w_e, 3)
        w_g1 = torch.cat((w_g, w_e),3)
        h_g = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
        h_e = x[:, :, 0, :] - x[:, :, h_x - 1, :]
        h_e = torch.unsqueeze(h_e, 2)
        h_g1 = torch.cat((h_g, h_e), 2)
        h_tv = (mask * torch.pow(h_g1, 2)).sum()
        #print(mask.size())
        #print(h_g1.size())
        w_tv = (mask * torch.pow(w_g1, 2)).sum()
        return (h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


#==================================>const loss<=================================
class const_loss(nn.Module):
    def __init__(self):
        super(const_loss, self).__init__()
        self.L_con = Myloss.L_con()
    def forward(self, x, neg, pos, k):
        return torch.mean(max(self.L_con(x, pos) - self.L_con(x, neg) + k, self.L_con(neg, x) - self.L_con(neg, x)))

class const_loss2(nn.Module):
    def __init__(self):
        super(const_loss2, self).__init__()
        self.L_con = Myloss.L_con()
        self.L_const = Myloss.PerceptualLoss()
    def forward(self, x, neg, pos, k):
        #print('ok')
        return torch.mean(max(self.L_const(x, pos) - self.L_const(x, neg) + k, self.L_con(neg, x) - self.L_con(neg, x))) # max(xx,0.000)

class Vgg_16(nn.Module):
    def __init__(self):
        super(Vgg_16, self).__init__()
        features = vgg16(pretrained=True).features.cuda()
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16,23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
       
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg_16().cuda()
        self.l1 = nn.L1Loss()
        self.weight = [1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss

if __name__ == "__main__":
    a = torch.randn((1, 3, 128, 128)).cuda()
    b = torch.randn((1, 3, 128, 128)).cuda()

    loss = GDLoss().cuda()
    c = loss(a)

    print(b)
    # print(c)
