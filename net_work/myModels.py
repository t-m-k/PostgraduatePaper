from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal_, constant_

from ResNetBlock import ResNetBlock,BasicBlock


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True):
        super(TSN, self).__init__()

        self.sobel = torch.nn.Conv2d(1, 1, 3, bias=False).cuda()
        self.sobel.weight.requires_grad = False
        self.sobel_x = torch.nn.Parameter(torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]))
        self.ReflectionPad = torch.nn.ReflectionPad2d(1).cuda()
        self.sobel_y = torch.nn.Parameter(torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]))
        self.sobel_x.requires_grad = False
        self.sobel_y.requires_grad = False

        self.model56to28 = ResNetBlock(BasicBlock, 4, inchannels=384).cuda()
        self.model28to14 = ResNetBlock(BasicBlock, 4, inchannels=768).cuda()
        self.model14to7 = ResNetBlock(BasicBlock, 4, inchannels=1152).cuda()
        self.avgpooling7to1 = nn.AvgPool2d(7, stride=1).cuda()
        self.fc1024to101 = nn.Linear(1152, 101).cuda()
        self.InnerConsensus = ConsensusModule(consensus_type)

        self.resnet_optimal = []

        self.resnet_subconv_stage1 = torch.nn.Conv2d(192, 128, 1).cuda()
        self.resnet_optimal.append(self.resnet_subconv_stage1)
        self.resnet_subconv_stage2 = torch.nn.Conv2d(256, 128, 1).cuda()
        self.resnet_optimal.append(self.resnet_subconv_stage2)
        self.resnet_subconv_stage3 = torch.nn.Conv2d(576, 128, 1).cuda()
        self.resnet_optimal.append(self.resnet_subconv_stage3)


        self.resnet_sobelconv_stage1 = torch.nn.Conv2d(192, 128, 1).cuda()
        self.resnet_optimal.append(self.resnet_sobelconv_stage1)
        self.resnet_sobelconv_stage2 = torch.nn.Conv2d(256, 128, 1).cuda()
        self.resnet_optimal.append(self.resnet_sobelconv_stage2)
        self.resnet_sobelconv_stage3 = torch.nn.Conv2d(576, 128, 1).cuda()
        self.resnet_optimal.append(self.resnet_sobelconv_stage3)

        self.resnet_conv3d_stage1 = nn.Conv3d(6, 1, 3, stride=1, padding=1).cuda()
        self.resnet_optimal.append(self.resnet_conv3d_stage1)
        self.resnet_conv3d_stage2 = nn.Conv3d(6, 1, 3, stride=1, padding=1).cuda()
        self.resnet_optimal.append(self.resnet_conv3d_stage2)
        self.resnet_conv3d_stage3 = nn.Conv3d(6, 1, 3, stride=1, padding=1).cuda()
        self.resnet_optimal.append(self.resnet_conv3d_stage3)

        for m in self.resnet_optimal:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # print('2d')
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # print('2d')

        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")


        self._construct_RGB_model(base_model,num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _construct_RGB_model(self,base_model,num_class):
        import tf_model_zoo
        self.RGB_base_model = getattr(tf_model_zoo, base_model)()
        self.RGB_base_model.last_layer_name = 'fc'
        self.RGB_input_size = 224
        self.RGB_input_mean = [104, 117, 128]
        self.RGB_input_std = [1]

        feature_dim = getattr(self.RGB_base_model, self.RGB_base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.RGB_base_model, self.RGB_base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.RGB_new_fc = None
        else:
            setattr(self.RGB_base_model, self.RGB_base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.RGB_new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.RGB_new_fc is None:
            normal_(getattr(self.RGB_base_model, self.RGB_base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.RGB_base_model, self.RGB_base_model.last_layer_name).bias, 0)
        else:
            normal_(self.RGB_new_fc.weight, 0, std)
            constant_(self.RGB_new_fc.bias, 0)



    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)

        # print(self.base_model)
        # print(self.new_fc)
        # exit()
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            # exit()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            count = 0
            for m in self.RGB_base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0

        for m in self.base_model.modules():
            # print(m)
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        conv_cnt = 0
        bn_cnt = 0
        for m in self.RGB_base_model.modules():
            # print(m)
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        # print(len(first_conv_weight))

        # exit()
        self.resnet_optimal.append(self.model56to28)
        self.resnet_optimal.append(self.model28to14)
        self.resnet_optimal.append(self.model14to7)
        self.resnet_optimal.append(self.fc1024to101)
        print(len(self.resnet_optimal))
        for mm in self.resnet_optimal:
            for m in mm.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])

                elif isinstance(m, torch.nn.BatchNorm1d):
                    bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm2d):
                    bn.extend(list(m.parameters()))
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError(
                            "New atomic module type: {}. Need to give it a learning policy".format(type(m)))


        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        # sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        RGB_sample_len = 3 * 1

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length

            RGB_input,input = self._get_diff(input)  # out is torch.Size([10, 7, 6, 3, 224, 224]) ====>out is torch.Size([10, 7, 5, 3, 224, 224])

            # print(input.size())
            # exit()

        # print(self.base_model)
        #print(self.RGB_base_model)
        # print(self.RGB_base_model.ReLU)
        # print(self.RGB_base_model.Concat_544)
        # exit()
        # 9            33         56      73       96      119      142      165      182      205     228
        # RGB_opid,RGB_base_out = self.RGB_base_model(RGB_input.view((-1, RGB_sample_len) + RGB_input.size()[-2:]),(0,9))
        # opid,base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]),(0,9))

        RGB_base_out = RGB_input.view((-1, RGB_sample_len) + RGB_input.size()[-2:])
        base_out = input.view((-1, sample_len) + input.size()[-2:])
        print(RGB_base_out.size())
        print(base_out.size())

        j = 0
        # for i in [9,33,56,73,96,119,142,165,182,205,228,230]:      #228 is the last concat layers
        ResBlockOut = 0
        ConThreeDout = 0

        # loop_num = -1
        # for i in [9,  56, 165,  228,230]:  #9 torch.Size([35, 192, 56, 56]) 33 torch.Size([35, 256, 28, 28]) 73 torch.Size([35, 576, 14, 14])
        for i in [9, 33, 73, 230]:
            RGB_opid, RGB_base_out = self.RGB_base_model(RGB_base_out,(j, i))
            opid,base_out = self.base_model(base_out,(j,i))
            if i == 230:
                break

            j = i+1

            att_mean = torch.Tensor.mean(base_out)
            att_var = torch.Tensor.var(base_out,False)
            att =torch.sigmoid((base_out-att_mean)/att_var)
            RGB_base_out = torch.mul(RGB_base_out,att)
            RGB_mean = torch.Tensor.mean(RGB_base_out)
            RGB_var = torch.Tensor.var(RGB_base_out, False)
            RGB_base_out = (RGB_base_out-RGB_mean)/RGB_var
            print(RGB_base_out.size())
            print(base_out.size())
            print(att_mean.size())
            print(att_var.size())
            print(att_var.size())
            exit()

            # print(i)
            # print(RGB_base_out.size())

            # print(RGB_base_out.size())


            # sub branch
            if j == 10:
                ResInput_preconv = self.resnet_subconv_stage1(RGB_base_out)
            if j == 34:
                ResInput_preconv = self.resnet_subconv_stage2(RGB_base_out)
            if j == 74:
                ResInput_preconv = self.resnet_subconv_stage3(RGB_base_out)

            ResInput = ResInput_preconv.view((-1, self.num_segments) + ResInput_preconv.size()[1:])
            ResInput_Sub = ResInput[:, 1:, :, :, :].clone()
            for x in reversed(list(range(1, ResInput.size()[1]))):
                ResInput_Sub[:, x - 1, :, :, :] = ResInput[:, x, :, :, :] - ResInput[:, x - 1, :, :, :]

            ResInput_Sub = ResInput_Sub.view((-1,ResInput_Sub.size()[2]) + ResInput_Sub.size()[3:])

            # sobel branch
            ResInput_Sobel = RGB_base_out.view((-1, RGB_base_out.size()[1]) + RGB_base_out.size()[2:])[
                             RGB_base_out.size()[0] - ResInput_Sub.size()[0]:, :, :, :]  # 5 = bathsize
            if j == 10:
                filtered_sobel_preconv = self.resnet_sobelconv_stage1(ResInput_Sobel)
            if j == 34:
                filtered_sobel_preconv = self.resnet_sobelconv_stage2(ResInput_Sobel)
            if j == 74:
                filtered_sobel_preconv = self.resnet_sobelconv_stage3(ResInput_Sobel)

            ResInput_Sobel = self.ReflectionPad(filtered_sobel_preconv)
            self.sobel.weight = self.sobel_x
            filtered_x = self.sobel(ResInput_Sobel[:, 0:1, :, :])
            for channal in range(ResInput_Sobel.size()[1]):
                if  channal != 0:
                    k = self.sobel(ResInput_Sobel[:, channal:channal + 1, :, :])
                    filtered_x = torch.cat((filtered_x, k), dim=1)

            self.sobel.weight = self.sobel_y
            filtered_y = self.sobel(ResInput_Sobel[:, 0:1, :, :])
            for channal in range(ResInput_Sobel.size()[1]):
                if channal != 0:
                    k = self.sobel(ResInput_Sobel[:, channal:channal + 1, :, :])
                    filtered_y = torch.cat((filtered_y, k), dim=1)

            filtered_sobel = torch.cat((filtered_x, filtered_y), dim=1)

            # print(ResInput_Sub.size())
            # print(filtered_sobel.size())
            # exit()

            filtered_out = torch.cat((ResInput_Sub, filtered_sobel), dim=1)

            # 3dconv branch
            if j == 10:
                ConThreeDout = self.resnet_conv3d_stage1(filtered_out.view((-1, 6) + filtered_out.size()[1:]))  # [bx1x384x56x56(28x28)]
            if j == 34:
                ConThreeDout = self.resnet_conv3d_stage1(filtered_out.view((-1, 6) + filtered_out.size()[1:]))  # [bx1x384x56x56(28x28)]
            if j == 74:
                ConThreeDout = self.resnet_conv3d_stage1(filtered_out.view((-1, 6) + filtered_out.size()[1:]))  # [bx1x384x56x56(28x28)]

            ConThreeDout = ConThreeDout.squeeze(1)

            # 9, 56, 165, 228, 230  #9, 33, 73, 230
            if j == 10:
                ResBlockOut = self.model56to28(ConThreeDout)
            if j == 34:
                ResBlockOut = self.model28to14(torch.cat((ResBlockOut, ConThreeDout), dim=1))
            if j == 74:
                ResBlockOut = self.model14to7(torch.cat((ResBlockOut, ConThreeDout), dim=1))
            # if j == 229:
            #     ResBlockOut = torch.cat((ResBlockOut, filtered_out), dim=1)#此处客家resnet

        ResBlockOut = self.avgpooling7to1(ResBlockOut)
        ResBlockOut = ResBlockOut.view(ResBlockOut.size(0), -1)
        ResBlockOut = self.fc1024to101(ResBlockOut)  #[bx101]
        # exit()


        # print(RGB_base_out[0][0][0])
        # print(RGB_base_out.size())
        # print(base_out[0][0][0])
        # print(base_out.size())
        # print(RGB_opid)
        # print(opid)
        # exit()

        if self.dropout > 0:
            base_out = self.new_fc(base_out)
            RGB_base_out = self.RGB_new_fc(RGB_base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
            RGB_base_out = self.softmax(RGB_base_out)
            ResBlockOut = self.softmax(ResBlockOut)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            RGB_base_out = RGB_base_out.view((-1, self.num_segments) + RGB_base_out.size()[1:])
        #     # ResBlockOut = ResBlockOut.view((-1, self.num_segments-1) + ResBlockOut.size()[1:])

        output = self.consensus(base_out)
        RGB_output = self.consensus(RGB_base_out)
        # # ResBlockOut = self.consensus(ResBlockOut)

        # print(output.size()) #torch.Size([7, 1, 101])
        # print(RGB_output.size()) #torch.Size([7, 1, 101])
        # print(ResBlockOut.size())  #torch.Size([7, 101])
        # exit()

        ResBlockOut = ResBlockOut.unsqueeze(1)
        myoutput = torch.cat((output, RGB_output,ResBlockOut), dim=1)
        # myoutput = torch.cat((output,RGB_output),dim = 1)
        # # print(myoutput.size())
        # # myoutput = myoutput.view(output.size(0),-1,output.size(1))
        # # print(myoutput.size())
        myoutput = self.consensus(myoutput)
        # print(myoutput.size())

        # inneroutput = torch.cat((myoutput.squeeze(1),ResBlockOut),dim = 1)
        # inneroutput = inneroutput.view(ResBlockOut.size(0),-1,ResBlockOut.size(1))
        # inneroutput = self.InnerConsensus(inneroutput)
        # return ResBlockOut  #.squeeze(1)
        return myoutput.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        # print(input.size())  # torch.Size([10, 126, 224, 224])

        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        RGB_input = input_view[:, :, 4:5, :, :, :].clone()
        # print(input_view.size())
        # print(self.num_segments)
        # print(self.new_length)
        # print(input_c)
        # exit(0)

        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return RGB_input,new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()

        # print(kernel_size)

        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            # print(new_kernel_size)
            # exit()
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
