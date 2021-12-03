import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Integral
from collections import OrderedDict

def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,  # weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias=False)

        self._batch_norm = nn.BatchNorm2d(
            out_channels)
        if act is None:
            self.act = nn.Identity()
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'hard_swish':
            self.act = nn.Hardswish(inplace=True)
        else:
            print(act)
            raise NotImplementedError('Wrong activation parameter!')

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self.act(self._batch_norm(y))
        return y


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True)
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs)
        return inputs * outputs


class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 act="relu"):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels)

        self._conv_linear = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

    def forward(self, inputs):
        x1, x2 = torch.split(
            inputs,
            [inputs.shape[1] // 2, inputs.shape[1] // 2],
            dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], dim=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 act="relu"):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=in_channels,
            act=None)
        self._conv_linear_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        # branch2
        self._conv_pw_2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels // 2)
        self._conv_linear_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=out_channels,
            act="hard_swish")
        self._conv_pw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act="hard_swish")

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)
        return out


class ESNet(nn.Module):
    def __init__(self,
                 scale=1.0,
                 act="hard_swish",
                 feature_maps=None,
                 channel_ratio=None):
        super(ESNet, self).__init__()
        if channel_ratio is None:
            channel_ratio = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        if feature_maps is None:
            feature_maps = [4, 11, 14]
        self.scale = scale
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]
        self.feature_maps = feature_maps
        stage_repeats = [3, 7, 3]

        stage_out_channels = [
            -1, 24, make_divisible(128 * scale), make_divisible(256 * scale),
            make_divisible(512 * scale), 1024
        ]

        self.out_channels = [24]
        self._feature_idx = 0
        # 1. conv1
        self._conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            act=act)
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._feature_idx += 1

        # 2. bottleneck sequences
        self._block_list = nn.ModuleList()
        arch_idx = 0
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                channels_scales = channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales),
                    divisor=8)
                if i == 0:
                    block = InvertedResidualDS(
                            in_channels=stage_out_channels[stage_id + 1],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=2,
                            act=act)
                else:
                    block = InvertedResidual(
                            in_channels=stage_out_channels[stage_id + 2],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=1,
                            act=act)
                self._block_list.append(block)
                arch_idx += 1
                self._feature_idx += 1
                self._update_out_channels(stage_out_channels[stage_id + 2],
                                          self._feature_idx, self.feature_maps)
        self.load_pretrained_weights()

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self.out_channels.append(channel)

    def load_pretrained_weights(self):
        ckpt = torch.load('/home/tjm/Documents/python/pycharmProjects/centerdet/samples/ESNet_x0_75_pretrained.pt')
        # new_dict = OrderedDict()
        for k, v in ckpt.items():
            v_shape = v.shape
            if len(v_shape) == 4:
                if v_shape[1] == 1 and v_shape[2] == 3:
                    ckpt[k] = F.pad(v, (1, 1, 1, 1), value=0)
            # new_dict[v]
        self.load_state_dict(ckpt, strict=True)
        del ckpt
        # print('loaded pretrained weights!')

    def forward(self, x):
        x = self._conv1(x)
        x = self._max_pool(x)
        outs = [x]
        for i, inv in enumerate(self._block_list):
            x = inv(x)
            if i + 2 in self.feature_maps:
                outs.append(x)
        return outs


if __name__ == '__main__':
#     from collections import OrderedDict
#     new_dit = OrderedDict()
#     inp = torch.randn((4, 3, 320, 320))
    m = ESNet(scale=0.75, channel_ratio=[0.875, 0.5, 0.5, 0.5, 0.625, 0.5, 0.625, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
#     stat = torch.load('/home/tjm/Documents/python/pycharmProjects/centerdet/samples/ESNet_x0_75_pretrained.pth')
#     # key_ls = list()
#
#     key_ls = list()
#     for k, v in m.state_dict().items():
#         print(k, v.shape)
#         if k.split('.')[-1] != 'num_batches_tracked':
#             key_ls.append(k)
#     print(len(key_ls))
#     # print(m.out_channels)
#     # for i in m(inp):
#     #     print(i.shape)
#     for idx, (k, v) in enumerate(stat.items()):
#         # print(k, v.shape)
#         new_dit[key_ls[idx]] = v
#
#     m.load_state_dict(new_dit, strict=True)
#     torch.save(m.state_dict(), '/home/tjm/Documents/python/pycharmProjects/centerdet/samples/ESNet_x0_75_pretrained.pt')