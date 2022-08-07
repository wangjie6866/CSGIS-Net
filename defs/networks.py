'''
implementations of network architectures,
where : HDC-edge-refine is the backbone of smoothing network,
        others : any change about the smoothing network
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = [
    "Baseline",
    "HDC_edge_refine",
    "HDC_att",
    "HDC_cbam",
    "PDP_edge_refine",
    "RDB_edge_refine",
    "Edge_guided",
    
]
class ResBlock(nn.Module):  # with 3 convs
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x0 = x
        x = self.triple_conv(x) + x0
        return x

class rHDCblock(nn.Module):
    def __init__(self, dim):
        super(rHDCblock, self).__init__()

        self.conv_3x3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(dim)

        self.conv_3x3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(dim)

        self.conv_3x3_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(dim)

    def forward(self, feature_map):

        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(out_3x3_1)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(out_3x3_2)))
        return out_3x3_3 + feature_map


class Descriptor_Res(nn.Module):
    def __init__(self, in_c, dim, num_block):
        super(Descriptor_Res, self).__init__()
        self.conv_in = nn.Conv2d(in_c, dim, kernel_size=3, padding=1)
        nets = []
        for i in range(num_block):
            nets.append(ResBlock(dim))
        self.res16 = nn.ModuleList(nets)

    def forward(self, x):
        x = self.conv_in(x)
        for m in self.res16:
            x = m(x)
        return x

class Descriptor_rHDC(nn.Module):
    def __init__(self, in_c, dim, num_block):
        super(Descriptor_rHDC, self).__init__()
        self.conv_in = nn.Conv2d(in_c, dim, kernel_size=3, padding=1)
        nets = []
        for i in range(num_block):
            nets.append(rHDCblock(dim))
        self.res16 = nn.ModuleList(nets)

    def forward(self, x):
        x = self.conv_in(x)
        for m in self.res16:
            x = m(x)
        return x

class Baseline(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(Baseline, self).__init__()
        self.descriptor = Descriptor_Res(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x = self.descriptor(x)
        x = self.interpreter(x)
        return x



class HDC_edge_refine(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDC_edge_refine, self).__init__()
        self.descriptor = Descriptor_rHDC(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        )
        self.interpreter2 = nn.Sequential(
            nn.Conv2d(dim+1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim+1+out_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.descriptor(x)
        edge = self.interpreter1(x)
        input_features1 = torch.cat((x, edge), dim=1)
        out1 = self.interpreter2(input_features1)
        input_features2 = torch.cat((input_features1, out1), dim=1)
        out2 = self.refine(input_features2)
        return edge, out1, out2



class HDC_att(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDC_att, self).__init__()
        self.descriptor = Descriptor_rHDC(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        )
        self.interpreter2 = nn.Sequential(
            nn.Conv2d(dim+1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim+1+out_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )

    def eca(self, x, gamma, b):
        N, C, H, W = x.size()

        t = int(abs((log(C, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        avg_pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)

        y = avg_pool(x)
        y = conv(y.squeeze(-1).transpose(-1,-2))
        y = y.transpose(-1,-2).unsqueeze(-1)

        return x * y.expand_as(x)

    def forward(self, x):
        x = self.descriptor(x)
        edge = self.interpreter1(x)
        input_features1 = torch.cat((x, edge), dim=1)
        input_features1 = self.eca(input_features1, 2, 1)
        out1 = self.interpreter2(input_features1)
        input_features2 = torch.cat((input_features1, out1), dim=1)
        input_features2 = self.eca(input_features2, 2, 1)
        out2 = self.refine(input_features2)
        return edge, out1, out2

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
        Flatten(),
        nn.Linear(gate_channels, gate_channels // reduction_ratio),
        nn.ReLU(),
        nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
            # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
    
    def logsumexp_2d(tensor):
        tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
        s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
        outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
        return outputs
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class HDC_cbam(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDC_cbam, self).__init__()
        self.descriptor = Descriptor_rHDC(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        )
        self.interpreter2 = nn.Sequential(
            nn.Conv2d(dim+1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim+1+out_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.cbam1 = CBAM(dim+1)
        self.cbam2 = CBAM(dim+1+out_c)



    def forward(self, x):
        x = self.descriptor(x)
        edge = self.interpreter1(x)
        input_features1 = torch.cat((x, edge), dim=1)
        #input_features1 = self.eca(input_features1, 2, 1)
        input_features1 = self.cbam1(input_features1)
        out1 = self.interpreter2(input_features1)
        input_features2 = torch.cat((input_features1, out1), dim=1)
        #input_features2 = self.eca(input_features2, 2, 1)
        input_features2 = self.cbam2(input_features2)
        out2 = self.refine(input_features2)
        return edge, out1, out2


class HDC_no_edge(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDC_edge_refine, self).__init__()
        self.descriptor = Descriptor_rHDC(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        )
        self.interpreter2 = nn.Sequential(
            nn.Conv2d(dim+1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim+1+out_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )

    def forward(self, x, seg):
        x = self.descriptor(x)
        #edge = self.interpreter1(x)
        input_features1 = torch.cat((x, seg), dim=1)
        out1 = self.interpreter2(input_features1)
        input_features2 = torch.cat((input_features1, out1), dim=1)
        out2 = self.refine(input_features2)
        return edge, out1, out2


class PPblock(nn.Module):
    def __init__(self, in_dim, out_dim,k):
        super(PPblock, self).__init__()

        self.conv_3x3_1 = nn.Conv2d(in_dim, out_dim, kernel_size=3,stride=1,padding=1**(k-1),dilation=1**(k-1))
        self.conv_3x3_2 = nn.Conv2d(in_dim, out_dim, kernel_size=3,stride=1,padding=2**(k-1),dilation=2**(k-1))
        self.conv_3x3_3 = nn.Conv2d(in_dim, out_dim, kernel_size=3,stride=1,padding=3**(k-1),dilation=3**(k-1))
        self.conv_3x3_4 = nn.Conv2d(in_dim, out_dim, kernel_size=3,stride=1,padding=4**(k-1),dilation=4**(k-1))

        self.conv_1x1_1 = nn.Conv2d(out_dim*4, out_dim, kernel_size=1,stride=1,padding=0)
        self.conv_1x1_2 = nn.Conv2d(out_dim*3, out_dim, kernel_size=1,stride=1,padding=0)
        self.conv_1x1_3 = nn.Conv2d(out_dim*2, out_dim, kernel_size=1,stride=1,padding=0)

        self.conv_1x1_0 = nn.Conv2d(out_dim*4, out_dim, kernel_size=1,stride=1,padding=0)#maybe try kernelsize3
        self.cbam = CBAM(out_dim)

    def forward(self, x):
        x_1 = self.conv_3x3_1(x)
        x_2 = self.conv_3x3_2(x)
        x_3 = self.conv_3x3_3(x)
        x_4 = self.conv_3x3_4(x)

        x_11 = self.conv_1x1_1(torch.cat((x_1,x_2,x_3,x_4), dim=1))
        x_22 = self.conv_1x1_2(torch.cat((x_2,x_3,x_4), dim=1))
        x_33 = self.conv_1x1_3(torch.cat((x_3,x_4), dim=1))

        x_cat = torch.cat((x_11,x_22,x_33,x_4),dim=1)
        x_out = self.conv_1x1_0(x_cat)
        x_out = self.cbam(x_out)

        return x_out

class PDPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super(PDPBlock, self).__init__()
        self.PP1 = PPblock(in_dim, out_dim, 1)
        self.PP2 = PPblock(in_dim * 2, out_dim, 2)
        self.PP3 = PPblock(in_dim * 3, out_dim, 3)
        self.PP4 = PPblock(in_dim * 4, out_dim, 4)

    def forward(self, x):
        #print(x.size())
        x_1 = self.PP1(x)
        x_2 = self.PP2(torch.cat((x,x_1),dim=1))
        x_3 = self.PP3(torch.cat((x,x_1,x_2),dim=1))
        x_4 = self.PP4(torch.cat((x,x_1,x_2,x_3),dim=1))
        return x_4

class Descriptor_PDP(nn.Module):
    def __init__(self, in_c, dim, num_block):
        super(Descriptor_PDP, self).__init__()
        self.conv_in = nn.Conv2d(in_c, dim, kernel_size=3, padding=1)
        nets = []
        for i in range(num_block):
            nets.append(PDPBlock(dim, dim, 4))
        self.des = nn.ModuleList(nets)

    def forward(self, x):
        x = self.conv_in(x)
        for m in self.des:
            x = m(x)
        return x


class PDP_edge_refine(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #num_block is for descriptor
        super(PDP_edge_refine, self).__init__()
        self.descriptor = Descriptor_rHDC(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        )
        self.interpreter2 = nn.Sequential(
            nn.Conv2d(dim+1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            PDPBlock(dim, dim, 4),
            #PDPBlock(dim, dim, 4),
            #PDPBlock(dim, dim, 4),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim+1+out_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            PDPBlock(dim, dim, 4),
            #PDPBlock(dim, dim, 4),
            #PDPBlock(dim, dim, 4),
            #rHDCblock(dim),
            #rHDCblock(dim),
            #rHDCblock(dim),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.descriptor(x)
        edge = self.interpreter1(x)
        input_features1 = torch.cat((x, edge), dim=1)
        out1 = self.interpreter2(input_features1)
        input_features2 = torch.cat((input_features1, out1), dim=1)
        out2 = self.refine(input_features2)
        return edge, out1, out2




#================>>   RDB   <<===================
class Residual_Block(nn.Module):
    def __init__(self, i_channel, o_channel, stride=1, downsample=None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=i_channel, out_channels=o_channel, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(o_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=o_channel, out_channels=o_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(o_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class  DetailedResidualBlock(nn.Module):
    def __init__(self, channels):
        super(DetailedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1), nn.ReLU()
        )
        # self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv1 = nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=3, dilation=3)
        self.conv3 = nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=5, dilation=5)
        self.conv4 = nn.Sequential(nn.Conv2d(channels*3, channels, (3, 3), stride=(1, 1), padding=1),
                                   Residual_Block(channels,channels),
                                   nn.ReLU())
    def forward(self, x):
        inputs = self.conv0(x)
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        x3 = self.conv3(inputs)
        catout = torch.cat((x1, x2, x3), 1)
        out = self.conv4(catout)
        return x + out

class Descriptor_RDB(nn.Module):
    def __init__(self, in_c, dim, num_block):
        super(Descriptor_RDB, self).__init__()
        self.conv_in = nn.Conv2d(in_c, dim, kernel_size=3, padding=1)
        nets = []
        for i in range(num_block):
            nets.append(DetailedResidualBlock(dim))
        self.des = nn.ModuleList(nets)
        #self.pdp = PDPBlock(dim,dim,3)

    def forward(self, x):
        x = self.conv_in(x)
        for m in self.des:
            x = m(x)
        #x = self.pdp(x)

        return x

class RDB_edge_refine(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(RDB_edge_refine, self).__init__()
        self.descriptor = Descriptor_RDB(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        )
        self.interpreter2 = nn.Sequential(
            nn.Conv2d(dim+1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim+1+out_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.descriptor(x)
        edge = self.interpreter1(x)
        input_features1 = torch.cat((x, edge), dim=1)
        out1 = self.interpreter2(input_features1)
        input_features2 = torch.cat((input_features1, out1), dim=1)
        out2 = self.refine(input_features2)
        return edge, out1, out2


class EGNL(nn.Module):
    def __init__(self, in_channels):
        super(EGNL, self).__init__()

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)



    def forward(self, x, edge_map):
        n, c, h, w = x.size()
        x_down = self.down(x)

        # [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)

        ### appearance relation map
        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)

        # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        Ra = F.softmax(torch.bmm(theta, phi), 2)


        ### depth relation map
        edge1 = F.interpolate(edge_map, size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners = True).view(n, 1, int(h / 4)*int(w / 4)).transpose(1,2)
        edge2 = F.interpolate(edge_map, size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners = True).view(n, 1, int(h / 8)*int(w / 8))

        # n, (h / 4) * (w / 4), (h / 8) * (w / 8)
        edge1_expand = edge1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        edge2_expand = edge2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))

        Rd = torch.min(edge1_expand / (edge2_expand + self.eps), edge2_expand / (edge1_expand + self.eps))

        Rd = F.softmax(Rd, 2)

        S = F.softmax(Ra * Rd, 2)


        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(S, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))



        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners = True)


class Edge_guided(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(Edge_guided, self).__init__()
        self.descriptor = Descriptor_rHDC(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        )
        self.interpreter2 = nn.Sequential(
            nn.Conv2d(dim+1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            #nn.BatchNorm2d(dim),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim+1+out_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.egnl = EGNL(dim)
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(dim, out_c, 1, 1, 0, 1, bias=False),
        )

    def forward(self, x):
        x = self.descriptor(x)
        edge = self.interpreter1(x)
        input_features1 = torch.cat((x, edge), dim=1)
        x = self.interpreter2(input_features1)
        x = self.egnl(x, edge)
        out1 = self.tail(x)
        input_features2 = torch.cat((input_features1, out1), dim=1)
        out2 = self.refine(input_features2)
        return edge, out1, out2
