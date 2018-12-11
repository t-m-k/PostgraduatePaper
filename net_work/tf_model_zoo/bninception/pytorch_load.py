import torch
from torch import nn
from .layer_factory import get_basic_layer, parse_expr
import torch.utils.model_zoo as model_zoo
import yaml


class BNInception(nn.Module):
    def __init__(self, model_path='tf_model_zoo/bninception/bn_inception.yaml', num_classes=101,
                       weight_url='https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth'):
        super(BNInception, self).__init__()

        manifest = yaml.load(open(model_path))
        # print(manifest)

        layers = manifest['layers']
        # print(layers)

        self._channel_dict = dict()

        self._op_list = list()
        for l in layers:
            # print(l)
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat':
                id, out_name, module, out_channel, in_name = get_basic_layer(l,
                                                                3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]],
                                                                             conv_bias=True)
                # print(id, out_name, module, out_channel, in_name)

                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            else:
                # print(l)
                # print(id)
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel

        self.load_state_dict(torch.utils.model_zoo.load_url(weight_url))
        # print(len(self._op_list))
        # num = -1
        # for op in self._op_list:
        #     num += 1
        #     # print(num)
        #     # print(op)
        #     # if op[0] == 'conv2_relu_3x3':
        #     #     print(op)
        #     #     print(num)
        #     # if op[1] == 'Concat':
        #     #     print(op)
        #     #     print(num)
        #     if num > 228:
        #         print(op)
        #         print(num)
        # exit()
        # self.process = ['conv2_relu_3x3','Concat','Concat','Concat','Concat','Concat','Concat','Concat','Concat','Concat','Concat']
        #                        9            33         56      73       96      119      142      165      182      205     228

    def forward(self, input,process = (0,230)):

        # conv2_relu_3x3 Concat...
        # process = process[0]
        # print(type(process))
        # print(process)
        # print(process[0])
        # print(process[1] + 1)
        # for i in self._op_list:
        #     print(i)

        data_dict = dict()
        # data_dict[self._op_list[0][-1]] = input
        data_dict[self._op_list[process[0]][-1]] = input

        def get_hook(name):

            def hook(m, grad_in, grad_out):
                print(name, grad_out[0].data.abs().mean())

            return hook

        # for op in self._op_list:
        for op in self._op_list[process[0]:process[1]+1]:

            # if op[0] == 'conv2_relu_3x3':
            #     data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
            #     print(op[2])
            #     print(data_dict[op[2]][0][0][0])
            #     print(data_dict[op[2]].size())
            #     return data_dict[op[2]]


            if op[1] != 'Concat' and op[1] != 'InnerProduct':
                data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
                # getattr(self, op[0]).register_backward_hook(get_hook(op[0]))
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].size())
                    raise
        # return op[0],data_dict[self._op_list[-1][2]] #返回key有问题

        return op[1], data_dict[self._op_list[process[1]][2]]


class InceptionV3(BNInception):
    def __init__(self, model_path='model_zoo/bninception/inceptionv3.yaml', num_classes=101,
                 weight_url='https://yjxiong.blob.core.windows.net/models/inceptionv3-cuhk-0e09b300b493bc74c.pth'):
        super(InceptionV3, self).__init__(model_path=model_path, weight_url=weight_url, num_classes=num_classes)
