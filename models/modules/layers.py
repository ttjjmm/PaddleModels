import torch
import torch.nn as nn



class ConvNormLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 norm_groups=32,
                 use_dcn=False,
                 bias_on=False,
                 lr_scale=1.,
                 freeze_norm=False,
                 initializer=Normal(
                     mean=0., std=0.01),
                 skip_quant=False,
                 dcn_lr_scale=2.,
                 dcn_regularizer=L2Decay(0.)):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn']

        if bias_on:
            bias_attr = ParamAttr(
                initializer=Constant(value=0.), learning_rate=lr_scale)
        else:
            bias_attr = False

        if not use_dcn:
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(
                    initializer=initializer, learning_rate=1.),
                bias_attr=bias_attr)
            if skip_quant:
                self.conv.skip_quant = True
        else:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(
                    initializer=initializer, learning_rate=1.),
                bias_attr=True,
                lr_scale=dcn_lr_scale,
                regularizer=dcn_regularizer,
                dcn_bias_regularizer=dcn_regularizer,
                dcn_bias_lr_scale=dcn_lr_scale,
                skip_quant=skip_quant)

        norm_lr = 0. if freeze_norm else 1.
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2D(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=norm_groups,
                num_channels=ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        return out















