import tf_model_zoo
from torch import nn
import torch
from torch.nn.init import normal_, constant_
from tensorboardX import SummaryWriter
from torch.autograd import Variable

class HeadNet(nn.Module):
    def __init__(self):
        super(HeadNet, self).__init__()
        self.conv7x7 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv7x7_bn =nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.branch_relu = nn.ReLU(inplace = True)

        self.maxp3x3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.conv3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self,input):
        x = self.conv7x7(input)
        x = self.conv7x7_bn(x)
        x = self.branch_relu(x)

        x = self.maxp3x3(x)

        x = self.conv3x3_reduce(x)
        x = self.conv3x3_reduce_bn(x)
        x = self.branch_relu(x)

        x = self.conv3x3(x)
        x = self.conv3x3_bn(x)
        x = self.branch_relu(x)

        x = self.maxp3x3(x)
        return x

class BNUnit(nn.Module):
    def __init__(self):
        super(BNUnit, self).__init__()
        self.left_branch_conv = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.left_branch_bn =nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.branch_relu = nn.ReLU(inplace = True)

        self.left_mid_reduce_conv = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.left_mid_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.left_mid_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.left_mid_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.right_mid_double_reduce_conv = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.right_mid_double_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.right_mid_double_conv1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.right_mid_double_bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.right_mid_double_conv2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.right_mid_double_bn2 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.right_avgp = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.right_conv = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.right_bn = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self,input):
        left = self.left_branch_conv(input)
        left = self.left_branch_bn(left)
        left = self.branch_relu(left)

        left_mid = self.left_mid_reduce_conv(input)
        left_mid = self.left_mid_reduce_bn(left_mid)
        left_mid = self.branch_relu(left_mid)
        left_mid = self.left_mid_conv(left_mid)
        left_mid = self.left_mid_bn(left_mid)
        left_mid = self.branch_relu(left_mid)

        right_mid = self.right_mid_double_reduce_conv(input)
        right_mid = self.right_mid_double_reduce_bn(right_mid)
        right_mid = self.branch_relu(right_mid)
        right_mid = self.right_mid_double_conv1(right_mid)
        right_mid = self.right_mid_double_bn1(right_mid)
        right_mid = self.branch_relu(right_mid)
        right_mid = self.right_mid_double_conv2(right_mid)
        right_mid = self.right_mid_double_bn2(right_mid)
        right_mid = self.branch_relu(right_mid)

        right = self.right_avgp(input)
        right = self.right_conv(right)
        right = self.right_bn(right)
        right = self.branch_relu(right)

        res = torch.cat((left,left_mid,right_mid,right),dim =1)
        return res

class BN_Descent_Unit(nn.Module):
    def __init__(self):
        super(BN_Descent_Unit, self).__init__()
        self.left_branch_reduce_conv = nn.Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1))
        self.left_branch_reduce_bn =nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.branch_relu = nn.ReLU(inplace = True)
        self.left_branch_conv = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.left_branch_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.mid_double_reduce_conv = nn.Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1))
        self.mid_double_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mid_double_conv1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.mid_double_bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mid_double_conv2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.mid_double_bn2 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.right_maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

    def forward(self,input):
        left = self.left_branch_reduce_conv(input)
        left = self.left_branch_reduce_bn(left)
        left = self.branch_relu(left)
        left = self.left_branch_conv(left)
        left = self.left_branch_bn(left)
        left = self.branch_relu(left)

        mid = self.mid_double_reduce_conv(input)
        mid = self.mid_double_reduce_bn(mid)
        mid = self.branch_relu(mid)
        mid = self.mid_double_conv1(mid)
        mid = self.mid_double_bn1(mid)
        mid = self.branch_relu(mid)
        mid = self.mid_double_conv2(mid)
        mid = self.mid_double_bn2(mid)
        mid = self.branch_relu(mid)

        right = self.right_maxp(input)

        res = torch.cat((left, mid, right), dim=1)
        return res


class AttentionNet(nn.Module):
    def __init__(self,convNet):
        super(AttentionNet, self).__init__()
        self.conv7x7 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv7x7_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.branch_relu = nn.ReLU(inplace=True)

        self.maxp3x3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.att56 = convNet(64, 64, 1)
        self.att28 = convNet(64, 64 + 192, 2)
        self.att14 = convNet(64 + 192, 64 + 192 + 192, 2)
        self.att7 = convNet(64 + 192 + 192, 64 + 192 + 192 + 192, 2)

    def forward(self,input):
        # if num == 1:
        x = self.conv7x7(input)
        x = self.conv7x7_bn(x)
        x = self.branch_relu(x)

        x = self.maxp3x3(x)

        x = self.att56(x)
        # elif num == 2:
        x = self.att28(x)
        # elif num == 3:
        x = self.att14(x)
        # elif num == 4:
        x = self.att7(x)

        return x


class ConvNet(nn.Module):
    def __init__(self,inchannels,outchannels,step):
        super(ConvNet,self).__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=(3, 3), stride=(step, step), padding=(1, 1))
        self.bn = nn.BatchNorm2d(outchannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BaseNet(nn.Module):
    def __init__(self,convNet):
        super(BaseNet,self).__init__()
        self.conv7x7 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv7x7_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.branch_relu = nn.ReLU(inplace=True)

        self.maxp3x3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.conv3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3x3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.c28one = convNet(128, 128+128, 2)
        self.c28two = convNet(128+128, 128+128 + 64, 1)
        self.c14one = convNet(128+128 + 64, 128+128 + 64 + 128, 2)
        self.c14two = convNet(128+128 + 64 + 128, 128+128 + 64 + 128 + 64, 1)
        self.c7one = convNet(128+128 + 64 + 128 + 64, 128+128 + 64 + 128 + 64 + 128, 2)
        self.c7two = convNet(128+128 + 64 + 128 + 64 + 128, 128+128 + 64 + 128 + 64 + 128 + 64, 1)

        self.avgp7x7 =nn.MaxPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=True)
        self.liner = nn.Linear(128+128 + 64 + 128 + 64 + 128 + 64,101)

    def forward(self, input):
        x = self.conv7x7(input)
        x = self.conv7x7_bn(x)
        x = self.branch_relu(x)

        x = self.maxp3x3(x)

        x = self.conv3x3_reduce(x)
        x = self.conv3x3_reduce_bn(x)
        x = self.branch_relu(x)

        x = self.conv3x3(x)
        x = self.conv3x3_bn(x)
        x = self.branch_relu(x)

        x = self.c28one(x)
        x = self.c28two(x)
        x = self.c14one(x)
        x = self.c14two(x)
        x = self.c7one(x)

        x = self.c7two(x)
        x = self.avgp7x7(x)

        x = x.view(x.size(0), -1)
        x = self.liner(x)
        x = nn.Softmax()(x)

        return x

class WholeNet(nn.Module):
    def __init__(self,convNet,num_segments):
        super(WholeNet,self).__init__()
        self.num_segments = num_segments
        #BaseNet
        self.conv7x7 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv7x7_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7x7 = nn.ReLU(inplace=True)

        self.maxp3x3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.conv3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3x3_reduce = nn.ReLU(inplace=True)

        self.conv3x3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3x3 = nn.ReLU(inplace=True)

        self.c28one = convNet(128, 128 + 128, 2)
        self.c28two = convNet(128 + 128, 128 + 128 + 64, 1)
        self.c14one = convNet(128 + 128 + 64, 128 + 128 + 64 + 128, 2)
        self.c14two = convNet(128 + 128 + 64 + 128, 128 + 128 + 64 + 128 + 64, 1)
        self.c7one = convNet(128 + 128 + 64 + 128 + 64, 128 + 128 + 64 + 128 + 64 + 128, 2)
        self.c7two = convNet(128 + 128 + 64 + 128 + 64 + 128, 128 + 128 + 64 + 128 + 64 + 128 + 64, 1)

        self.avgp7x7 = nn.MaxPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=True)
        self.liner = nn.Linear(128 + 128 + 64 + 128 + 64 + 128 + 64, 101)

        # AttentionNet
        self.attconv7x7 = nn.Conv2d(15, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.attconv7x7_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.att_relu = nn.ReLU(inplace=True)

        self.attmaxp3x3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.att56 = convNet(64, 64, 1)
        self.att28 = convNet(64, 64 + 192, 2)
        self.att14 = convNet(64 + 192, 64 + 192 + 192, 2)
        self.att7 = convNet(64 + 192 + 192, 64 + 192 + 192 + 192, 2)

        # 3DRes
        self.resnet_subconv_stage1 = torch.nn.Conv2d(128, 128, 1)
        self.resnet_subconv_stage2 = torch.nn.Conv2d(320, 128, 1)
        self.resnet_subconv_stage3 = torch.nn.Conv2d(704, 128, 1)

        self.resnet_sobelconv_stage1 = torch.nn.Conv2d(128, 128, 1)
        self.resnet_sobelconv_stage2 = torch.nn.Conv2d(320, 128, 1)
        self.resnet_sobelconv_stage3 = torch.nn.Conv2d(704, 128, 1)

        self.resnet_conv3d_stage1 = nn.Conv3d(6, 1, 3, stride=1, padding=1)
        self.resnet_conv3d_stage2 = nn.Conv3d(6, 1, 3, stride=1, padding=1)
        self.resnet_conv3d_stage3 = nn.Conv3d(6, 1, 3, stride=1, padding=1)

        self.sobel = torch.nn.Conv2d(1, 1, 3, bias=False)
        self.sobel.weight.requires_grad = False
        self.sobel_x = torch.nn.Parameter(torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]))
        self.ReflectionPad = torch.nn.ReflectionPad2d(1)
        self.sobel_y = torch.nn.Parameter(torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]))
        self.sobel_x.requires_grad = False
        self.sobel_y.requires_grad = False
    def res3d(self,x):

        ## sub
        res3d_sub = self.resnet_subconv_stage1(x)
        res3d_sub = res3d_sub.view((-1, self.num_segments) + res3d_sub.size()[1:])
        ResInput_Sub = res3d_sub[:, 1:, :, :, :].clone()
        for x in reversed(list(range(1, res3d_sub.size()[1]))):
            ResInput_Sub[:, x - 1, :, :, :] = res3d_sub[:, x, :, :, :] - res3d_sub[:, x - 1, :, :, :]
        ResInput_Sub = ResInput_Sub.view((-1, ResInput_Sub.size()[2]) + ResInput_Sub.size()[3:])
        ## sob
        ResInput_Sobel = x[x.size()[0] - ResInput_Sub.size()[0]:, :, :, :]
        ResInput_Sobel = self.resnet_sobelconv_stage1(ResInput_Sobel)
        ResInput_Sobel = self.ReflectionPad(ResInput_Sobel)
        self.sobel.weight = self.sobel_x
        filtered_x = self.sobel(ResInput_Sobel[:, 0:1, :, :])
        for channal in range(ResInput_Sobel.size()[1]):
            if channal != 0:
                k = self.sobel(ResInput_Sobel[:, channal:channal + 1, :, :])
                filtered_x = torch.cat((filtered_x, k), dim=1)
        self.sobel.weight = self.sobel_y
        filtered_y = self.sobel(ResInput_Sobel[:, 0:1, :, :])
        for channal in range(ResInput_Sobel.size()[1]):
            if channal != 0:
                k = self.sobel(ResInput_Sobel[:, channal:channal + 1, :, :])
                filtered_y = torch.cat((filtered_y, k), dim=1)
        filtered_sobel = torch.cat((filtered_x, filtered_y), dim=1)
        filtered_out = torch.cat((ResInput_Sub, filtered_sobel), dim=1)
        ConThreeDout = self.resnet_conv3d_stage1(filtered_out.view((-1, 6) + filtered_out.size()[1:]))

        res3d_sob = self.resnet_sobelconv_stage1(x)
    def forward(self, input):
        # print(input.size())
        RGBinput = input[:,:3,:,:]
        Diffinput = input[:,3:,:,:]
        #basenet
        x = self.conv7x7(RGBinput)
        x = self.conv7x7_bn(x)
        x = self.relu7x7(x)
        x = self.maxp3x3(x)
        x = self.conv3x3_reduce(x)
        x = self.conv3x3_reduce_bn(x)
        x = self.relu3x3_reduce(x)

        #attention
        att = self.attconv7x7(Diffinput)
        att = self.attconv7x7_bn(att)
        att = self.att_relu(att)
        att = self.attmaxp3x3(att)
        att = self.att56(att)


        # attention56
        att_mean = torch.Tensor.mean(att)
        att_var = torch.Tensor.var(att, False)
        att = torch.sigmoid((att - att_mean) / att_var)
        x = torch.mul(x, att)
        x_mean = torch.Tensor.mean(x)
        x_var = torch.Tensor.var(x, False)
        x = (x - x_mean) / x_var

        ## afterattconv
        x = self.conv3x3(x)
        x = self.conv3x3_bn(x)
        x = self.relu3x3(x)
        ### 3dres
        self.res3d(x)

        x = self.c28one(x)

        att = self.att28(att)
        # attention28
        att_mean = torch.Tensor.mean(att)
        att_var = torch.Tensor.var(att, False)
        att = torch.sigmoid((att - att_mean) / att_var)
        x = torch.mul(x, att)
        x_mean = torch.Tensor.mean(x)
        x_var = torch.Tensor.var(x, False)
        x = (x - x_mean) / x_var

        ## afterattconv
        x = self.c28two(x)
        x = self.c14one(x)

        att = self.att14(att)
        # attention14
        att_mean = torch.Tensor.mean(att)
        att_var = torch.Tensor.var(att, False)
        att = torch.sigmoid((att - att_mean) / att_var)
        x = torch.mul(x, att)
        x_mean = torch.Tensor.mean(x)
        x_var = torch.Tensor.var(x, False)
        x = (x - x_mean) / x_var
        ## afterattconv
        x = self.c14two(x)
        x = self.c7one(x)

        att = self.att7(att)
        # attention7
        att_mean = torch.Tensor.mean(att)
        att_var = torch.Tensor.var(att, False)
        att = torch.sigmoid((att - att_mean) / att_var)
        x = torch.mul(x, att)
        x_mean = torch.Tensor.mean(x)
        x_var = torch.Tensor.var(x, False)
        x = (x - x_mean) / x_var
        ## afterattconv
        x = self.c7two(x)

        x = self.avgp7x7(x)

        x = x.view(x.size(0), -1)
        x = self.liner(x)
        x = nn.Softmax()(x)

        return x


dummy_input =  Variable(torch.randn(1,3+15,224,224))

WholeNet = WholeNet(ConvNet)
with SummaryWriter(comment='WholeNet') as w:
     w.add_graph(WholeNet, (dummy_input,))



# base_model = getattr(tf_model_zoo, 'BNInception')()
#
# base_model.last_layer_name = 'fc'
# input_size = 224
# input_mean = [104, 117, 128]
# input_std = [1]
#
# input_mean = input_mean * (1 + 5)
#
# num_class = 101
#
# feature_dim = getattr(base_model, base_model.last_layer_name).in_features
# setattr(base_model, base_model.last_layer_name, nn.Dropout(p=0.8))
# new_fc = nn.Linear(feature_dim, num_class)
#
# std = 0.001
#
# normal_(new_fc.weight, 0, std)
# constant_(new_fc.bias, 0)
#
#
# # dummy_input =  Variable(torch.randn(1,3,224,224))
# n = 0
# for m in base_model.modules():
#     n += 1
#     print(m)
#     # print(type(m))
# print(n)
# print(len(base_model.modules()[0]))

# print(base_model)
# print(dummy_input)

# with SummaryWriter(comment='BNInception') as w:
#      w.add_graph(base_model, (dummy_input,))