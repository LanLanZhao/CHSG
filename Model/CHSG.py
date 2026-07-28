import torch
from torch import nn
from timm.layers import DropPath
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
"""


class Process(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(Process, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(output_dim)

        self.shortcut = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.bn_shortcut = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.bn_shortcut(self.shortcut(residual))
        return F.gelu(out)


# class Process(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Process, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)
#         self.bn1 = nn.BatchNorm2d(output_dim)
#
#     def forward(self, x):
#         out = F.gelu(self.bn1(self.conv1(x)))
#         return out


class HSGA(nn.Module):
    """
    Hierarchical Sparse-Graph Attention
    """

    def __init__(self, in_channels, out_channels, tag, use_max=False, boundary_mode="cyclic"):
        super().__init__()
        self.tag = tag
        if boundary_mode not in ("cyclic", "zero", "reflect", "replicate"):
            raise ValueError(f"Unsupported boundary_mode: {boundary_mode}")
        self.boundary_mode = boundary_mode
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.use_max = use_max

        self.K = 2
        self.K_ = [1, 2]
        # 121  124

    def _shift(self, x, shift_h=0, shift_w=0):
        if self.boundary_mode == "cyclic":
            if shift_h:
                x = torch.roll(x, shifts=shift_h, dims=2)
            if shift_w:
                x = torch.roll(x, shifts=shift_w, dims=3)
            return x

        B, C, H, W = x.shape
        pad_top = max(shift_h, 0)
        pad_bottom = max(-shift_h, 0)
        pad_left = max(shift_w, 0)
        pad_right = max(-shift_w, 0)
        pad = (pad_left, pad_right, pad_top, pad_bottom)

        if self.boundary_mode == "zero":
            padded = F.pad(x, pad, mode="constant", value=0.0)
        else:
            padded = F.pad(x, pad, mode=self.boundary_mode)

        start_h = max(-shift_h, 0)
        start_w = max(-shift_w, 0)
        return padded[:, :, start_h:start_h + H, start_w:start_w + W]

    def forward(self, x):
        B, C, H, W = x.shape
        '''
        This is the 5 connection graph construction
        '''
        x_j = x - x

        if self.tag == 'local':
            for i in self.K_:  # 5*5的密集连接
                x_c1 = torch.roll(x, shifts=-i, dims=2)  # 上
                x_c2 = torch.roll(x, shifts=i, dims=2)  # 下
                x_r1 = torch.roll(x, shifts=-i, dims=3)  # 左
                x_r2 = torch.roll(x, shifts=i, dims=3)  # 右
                x_ul = torch.roll(torch.roll(x, shifts=-i, dims=2), shifts=-i, dims=3)  # 左上
                x_ur = torch.roll(torch.roll(x, shifts=-i, dims=2), shifts=i, dims=3)  # 右上
                x_dl = torch.roll(torch.roll(x, shifts=i, dims=2), shifts=-i, dims=3)  # 左下
                x_dr = torch.roll(torch.roll(x, shifts=i, dims=2), shifts=i, dims=3)  # 右下

                x_edge = [(x_c1 - x), (x_c2 - x), (x_r1 - x), (x_r2 - x), (x_ul - x), (x_ur - x), (x_dl - x),
                          (x_dr - x)]
                # 图差分
                x_edge_sum = torch.stack(x_edge).sum(dim=0)  # 先堆叠再求和
                x_j += x_edge_sum / 8.
        else:
            for k in range(self.K, H, self.K):  # 往外的稀疏连接, 上下左右轴的
                x_c = torch.cat([x[:, :, -k:, :], x[:, :, :-k, :]], dim=2)
                delta = (x_c - x)
                x_j = torch.max(x_j, delta)

                x_c = torch.cat([x[:, :, k:, :], x[:, :, :k, :]], dim=2)
                delta = (x_c - x)
                x_j = torch.max(x_j, delta)

                x_r = torch.cat([x[:, :, :, -k:], x[:, :, :, :-k]], dim=3)
                delta = (x_r - x)
                x_j = torch.max(x_j, delta)

                x_r = torch.cat([x[:, :, :, k:], x[:, :, :, :k]], dim=3)
                delta = (x_r - x)
                x_j = torch.max(x_j, delta)

        # 拼接 & 输出
        x_cat = torch.cat([x, x_j], dim=1)  # [B, 2C, H, W]
        out = self.nn(x_cat)
        # 融合中心注意
        return out


class HSGABoundary(HSGA):
    def forward(self, x):
        B, C, H, W = x.shape
        x_j = x - x

        if self.tag == 'local':
            for i in self.K_:
                neighbors = [
                    self._shift(x, shift_h=-i),
                    self._shift(x, shift_h=i),
                    self._shift(x, shift_w=-i),
                    self._shift(x, shift_w=i),
                    self._shift(x, shift_h=-i, shift_w=-i),
                    self._shift(x, shift_h=-i, shift_w=i),
                    self._shift(x, shift_h=i, shift_w=-i),
                    self._shift(x, shift_h=i, shift_w=i),
                ]
                x_edge_sum = torch.stack([neighbor - x for neighbor in neighbors]).sum(dim=0)
                x_j += x_edge_sum / 8.
        else:
            for k in range(self.K, H, self.K):
                for shifted in (
                    self._shift(x, shift_h=k),
                    self._shift(x, shift_h=-k),
                    self._shift(x, shift_w=k),
                    self._shift(x, shift_w=-k),
                ):
                    delta = shifted - x
                    x_j = torch.max(x_j, delta)

        x_cat = torch.cat([x, x_j], dim=1)
        out = self.nn(x_cat)
        return out


class Grapher(nn.Module):
    """ 图模块 """

    def __init__(self, in_channels, tag, rg=5, patch_size=13, boundary_mode="cyclic"):
        super().__init__()
        self.channels = in_channels
        self.CAPE = CAPE(in_channels, rg)  # CAPE(in_channels=in_channels, kernel_size=rg)  #

        self.hsga = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            HSGABoundary(in_channels, in_channels, tag=tag, boundary_mode=boundary_mode),
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        x = self.CAPE(x)
        shot_cut = x
        x = self.hsga(x) + shot_cut
        return x


class CAPE(nn.Module):
    """
    Full-Name: Center-Aware Positional Encoding
    Implementation of conditional positional encoding. For more details refer to paper:
    Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>_
    """

    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=in_channels
        )
        self.lamda1 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.lamda2 = nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_mid = x[:, :, H // 2: H // 2 + 1, W // 2: W // 2 + 1].repeat(1, 1, H, W)
        center_mid = -((abs(x - x_mid)) ** 2)
        center_mid = torch.sigmoid(center_mid) * 2  # ---------------------------------------- * 2 ?
        x = self.lamda1 * self.pe(x) + self.lamda2 * center_mid + x
        return x


class DCGraphBlock(nn.Module):
    """
    动态光谱维度方向建图
    """

    def __init__(self, in_dim, patch_size, drop_path=0., layer_scale_init_value=1e-5, boundary_mode="cyclic"):
        super().__init__()
        self.mixer_edge = Grapher(in_dim, rg=5, tag='local', patch_size=patch_size, boundary_mode=boundary_mode)

        self.ffn_edge = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim),
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
        )

        # 添加下采样池化层：空间减半，通道增一倍
        self.down_pooling = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_dim, in_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 2),
            nn.GELU()
        )

        # 由于通道数增加了一倍，需要调整第二个图模块的输入维度
        self.mixer_edge2 = Grapher(in_dim * 2, rg=5, tag='global', patch_size=patch_size // 2,
                                   boundary_mode=boundary_mode)
        # no Global   no local
        self.ffn_edge2 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim * 2, kernel_size=3, stride=1, padding=1, groups=in_dim * 2),
            nn.BatchNorm2d(in_dim * 2),
            nn.Conv2d(in_dim * 2, in_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 2),
            nn.GELU(),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(in_dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(in_dim), requires_grad=True)
        self.layer_scale_3 = nn.Parameter(layer_scale_init_value * torch.ones(in_dim * 2), requires_grad=True)
        self.layer_scale_4 = nn.Parameter(layer_scale_init_value * torch.ones(in_dim * 2), requires_grad=True)

    # local  global
    def forward(self, x):
        shortcut = x
        # 第一个图模块
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer_edge(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn_edge(x))

        # 下采样并增加通道数
        x = self.down_pooling(x + shortcut)

        shortcut = x
        x = x + self.drop_path(self.layer_scale_3.unsqueeze(-1).unsqueeze(-1) * self.mixer_edge2(x))
        x = x + self.drop_path(self.layer_scale_4.unsqueeze(-1).unsqueeze(-1) * self.ffn_edge2(x))
        return x + shortcut


class SSJM_2DD(nn.Module):
    def __init__(self, in_channels):
        super(SSJM_2DD, self).__init__()
        out_channels = in_channels
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out1 = self.conv1x1(x)
        out3 = self.conv3x3(x)
        out5 = self.conv5x5(x)
        out = self.fuse(out1 + out3 + out5)
        return out


class SSJM_MSEF(nn.Module):
    def __init__(self, dim, scale=[1, 3, 5, 7]):
        super().__init__()

        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels, kernel_size=scale[i], padding=scale[i] // 2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class SSJM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        self.branch_spa_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        self.branch_spa_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        self.branch_spa_3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.fuse = nn.Sequential(
            # 通道压缩
            nn.Conv2d(3 * in_channels, in_channels * 3, kernel_size=1),  # 3C -> C
            nn.BatchNorm2d(in_channels * 3),
            nn.GELU(),
            # 深度卷积
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels * 3),
            nn.GELU(),
            # 最终调整
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.proj = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # shortcut = x
        x_reshaped = x.view(B, C, H * W)

        b1 = self.branch1(x_reshaped)
        b1 = self.branch_spa_1(b1.view(B, C, H, W))
        b2 = self.branch2(x_reshaped)
        b2 = self.branch_spa_2(b2.view(B, C, H, W))
        b3 = self.branch3(x_reshaped)
        b3 = self.branch_spa_3(b3.view(B, C, H, W))

        fused = torch.cat([b1, b2, b3], dim=1)
        fused = fused.view(B, 3 * C, H, W)

        out = self.fuse(fused)
        out = self.proj(out.unsqueeze(dim=1))
        out = out.squeeze(dim=1)
        return out


class CHSG(torch.nn.Module):
    """
    """

    def __init__(self, channels, band, dropout=0., drop_path=0., num_classes=16, patch_size=13,
                 boundary_mode="cyclic"):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path, 2)]

        self.Graph_PRE = Process(input_dim=band, output_dim=channels)
        self.SSJM_PRE = Process(input_dim=band, output_dim=channels)  # Process(input_dim=band, output_dim=channels)

        self.BatchOne = SSJM(channels) # SSJM(channels)  #
        #
        self.BatchTwo = []
        self.BatchTwo.append(nn.Sequential(DCGraphBlock(channels, patch_size, drop_path=dpr[1],
                                                        boundary_mode=boundary_mode)))
        # ALL SSJM  CAPE  HSGA
        self.BatchTwo = nn.Sequential(*self.BatchTwo)

        self.SSJM_Pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), )
        self.GRAPH_Pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Sequential(nn.Linear(channels, num_classes, bias=False), )
        self.model_init()

        self.gama = nn.Parameter(0.5 * torch.ones(channels), requires_grad=True)

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        X = self.Graph_PRE(inputs)
        Y = self.SSJM_PRE(inputs)
        B, C, H, W = X.shape

        ResX = self.BatchTwo(X)  # B C H W  动态建图分支
        ResY = self.BatchOne(Y)  # B C H W  多尺度分支

        out1 = self.GRAPH_Pooling(ResX).squeeze(-1).squeeze(-1)
        out2 = self.SSJM_Pooling(ResY).squeeze(-1).squeeze(-1)
        out = out1 * self.gama + out2 * (1. - self.gama)
        out = self.head(out)
        return out


def build_model(num_classes, band, patch_size, boundary_mode="cyclic"):
    return CHSG(
        channels=64,
        band=band,
        drop_path=0.1,
        num_classes=num_classes,
        patch_size=patch_size,
        boundary_mode=boundary_mode,
    )
