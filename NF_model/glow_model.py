import math


import torch
import torch.nn as nn
import numpy as np


from .glow_modules import (
    Conv2d,
    Conv2dZeros,
    ActNorm2d,
    InvertibleConv1x1,
    Permute2d,
    LinearZeros,
    SqueezeLayer,
    Split2d,
    gaussian_likelihood,
    gaussian_sample,
)
from .glow_utils import split_feature, uniform_binning_correction


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)

        self.input_drop = nn.Dropout(p=0.2)

        self.act = nn.SELU(inplace=True)

        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs,
        '''
        x = self.input_drop(x)

        att_map = self._derive_att_map(x)

        x = self._project(x, att_map)

        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs,
        out_shape   :(#bs,
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs,
        out_shape   :(#bs,
        '''
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)

        att_map = att_map / self.temp

        att_map = nn.functional.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

class HtrgGraphAttentionLayer(nn.Module):
    """
    Core heterogeneous graph interaction block used by AASIST.
    """
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)

        self.input_drop = nn.Dropout(p=0.2)

        self.act = nn.SELU(inplace=True)

        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs,
        x2  :(#bs,
        '''
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)

        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs,
        out_shape   :(#bs,
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs,
        out_shape   :(#bs,
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        att_map = att_map / self.temp

        att_map = nn.functional.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs,
        out_shape   :(#bs,
        '''
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        att_map = att_map / self.temp

        att_map = nn.functional.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

class STGraphBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, temp=100.0):
        super().__init__()

        filts = [128, [1, 32], [32, 32], [32, 64], [in_channels, in_channels]]
        gat_dims = [128, 128]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        self.attention = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=(1, 1)),
        )

        self.pos_S = nn.Parameter(torch.randn(1, 42//2, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.output_proj = Conv2dZeros(in_channels, out_channels)

    def forward(self, x):
        B, _, F, T = x.shape

        w = self.attention(x)

        w1 = nn.functional.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        out_S = self.GAT_layer_S(e_S)

        w2 = nn.functional.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        out_T = self.GAT_layer_T(e_T)

        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        out_S = out_S.unsqueeze(1).repeat(1, T, 1, 1).permute(0, 3, 2, 1)
        out_T = out_T.unsqueeze(2).repeat(1, 1, F, 1).permute(0, 3, 2, 1)

        master_bias = master.squeeze(1)
        master_bias = master_bias.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, T)

        out = x + out_S + out_T + master_bias
        out = self.output_proj(out)

        return out

def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block

class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        if flow_coupling == "additive":
            self.block = get_block(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.block = get_block(in_channels // 2, in_channels, hidden_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        z, logdet = self.flow_permutation(z, logdet, False)

        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)

            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0) + 1e-6
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        z, logdet = self.flow_permutation(z, logdet, True)

        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet

class FlowNet(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        C, F, T = image_shape

        for i in range(L):
            C, F, T = C * 4, F//2, T//2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, F, T])

            for _ in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                    )
                )
                self.output_shapes.append([-1, C, F, T])

            if i < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, F, T])
                C = C // 2

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        logdet = torch.zeros(z.shape[0], device=z.device)
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z

class Glow(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        learn_top,
        R,
    ):
        super().__init__()

        self.flow = FlowNet(
            image_shape=image_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
        )
        self.R = R
        self.learn_top = learn_top

        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.flow.output_shapes[-1][1] * 2,
                    self.flow.output_shapes[-1][2],
                    self.flow.output_shapes[-1][3],
                ]
            ),
        )
        self.register_buffer(
            "prior_h_normal",
            torch.concat(
                (
                    torch.ones([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                self.flow.output_shapes[-1][3]]) * self.R,

                    torch.zeros([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                 self.flow.output_shapes[-1][3]]),
                ), dim=0
            ))
        self.register_buffer(
            "prior_h_abnormal",
            torch.concat(
                (
                    torch.ones([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                self.flow.output_shapes[-1][3]]) * self.R * -1,

                    torch.zeros([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                 self.flow.output_shapes[-1][3]]),
                ), dim=0
            ))

    def prior(self, data, label=None):
        if data is not None:
            if label is not None:
                h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
                h[label == 0] = self.prior_h_normal
                h[label == 1] = self.prior_h_abnormal
            else:
                h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            h = self.prior_h_normal.repeat(32, 1, 1, 1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        return split_feature(h, "split")

    def forward(self, x=None, y_onehot=None):
        b, c, h, w = x.shape

        z, objective = self.flow(x, reverse=False)

        mean, logs = self.prior(x, y_onehot)

        objective += gaussian_likelihood(mean, logs, z)

        nll = (-objective) / (math.log(2.0) * c * h * w)

        return z, nll

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True

class GlobalFlowNetNoSqueeze(nn.Module):
    """
    Global flow stack without squeeze or split operations.
    """
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()
        C, F, T = image_shape
        self.layers = nn.ModuleList()
        self.output_shapes = []

        for _ in range(K):
            self.layers.append(
                FlowStep(
                    in_channels=C,
                    hidden_channels=hidden_channels,
                    actnorm_scale=actnorm_scale,
                    flow_permutation=flow_permutation,
                    flow_coupling=flow_coupling,
                    LU_decomposed=LU_decomposed,
                )
            )
            self.output_shapes.append([-1, C, F, T])

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        logdet = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            z, logdet = layer(z, logdet=0, reverse=True)
        return z

class MultiBandGlowC(nn.Module):
    """
    Multi-band flow model with band-wise flows followed by a global prior.
    """

    def __init__(
        self,
        image_shape=(32, 42, 202),
        n_bands=2,
        hidden_channels=256,
        K=8,
        L=1,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="affine",
        LU_decomposed=True,
        learn_top=False,
        R=1.0,
        global_K=None,
        global_hidden_channels=None,
    ):
        super().__init__()
        C, F, T = image_shape
        assert F % n_bands == 0, f"F={F} must be divisible by n_bands={n_bands}"
        self.C, self.F, self.T = C, F, T
        self.n_bands = n_bands
        self.band_F = F // n_bands

        self.band_flows = nn.ModuleList()
        for _ in range(n_bands):
            self.band_flows.append(
                FlowNet(
                    image_shape=(C, self.band_F, T),
                    hidden_channels=hidden_channels,
                    K=K,
                    L=L,
                    actnorm_scale=actnorm_scale,
                    flow_permutation=flow_permutation,
                    flow_coupling=flow_coupling,
                    LU_decomposed=LU_decomposed,
                )
            )

        outC = self.band_flows[0].output_shapes[-1][1]
        outF = self.band_flows[0].output_shapes[-1][2]
        outT = self.band_flows[0].output_shapes[-1][3]
        self.band_latent_shape = (outC, outF, outT)

        global_C = outC * n_bands
        global_F = outF
        global_T = outT

        if global_K is None:
            global_K = K
        if global_hidden_channels is None:
            global_hidden_channels = hidden_channels

        self.global_flow = GlobalFlowNetNoSqueeze(
            image_shape=(global_C, global_F, global_T),
            hidden_channels=global_hidden_channels,
            K=global_K,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
        )

        self.learn_top = learn_top
        self.R = R

        if learn_top:
            self.learn_top_fn = Conv2dZeros(global_C * 2, global_C * 2)

        self.register_buffer("prior_h", torch.zeros([1, global_C * 2, global_F, global_T]))

        self.register_buffer(
            "prior_h_normal",
            torch.concat(
                (
                    torch.ones([global_C, global_F, global_T]) * self.R,
                    torch.zeros([global_C, global_F, global_T]),
                ),
                dim=0,
            ),
        )
        self.register_buffer(
            "prior_h_abnormal",
            torch.concat(
                (
                    torch.ones([global_C, global_F, global_T]) * (-self.R),
                    torch.zeros([global_C, global_F, global_T]),
                ),
                dim=0,
            ),
        )

    def prior(self, batch_size, device, label=None):
        h = self.prior_h.repeat(batch_size, 1, 1, 1).to(device)

        if label is not None:
            h[label == 0] = self.prior_h_normal
            h[label == 1] = self.prior_h_abnormal

        if self.learn_top:
            h = self.learn_top_fn(h)

        return split_feature(h, "split")

    def forward(self, x, y=None):
        """
        x: (B, 32, 42, 202)
        y: Optional class labels with shape `(B,)`.
        """
        B, C, F, T = x.shape

        assert (C, F, T) == (self.C, self.F, self.T), f"Input shape mismatch: got {(C, F, T)}"

        logdet_sum = torch.zeros(B, device=x.device)
        z_list = []

        for i, flow in enumerate(self.band_flows):
            f0 = i * self.band_F
            f1 = (i + 1) * self.band_F
            x_band = x[:, :, f0:f1, :]

            z_band, logdet_band = flow(x_band, reverse=False)
            logdet_sum = logdet_sum + logdet_band
            z_list.append(z_band)

        z_cat = torch.cat(z_list, dim=1)

        z, logdet_g = self.global_flow(z_cat, reverse=False)
        logdet_sum = logdet_sum + logdet_g

        mean, logs = self.prior(B, x.device, label=y)
        objective = logdet_sum + gaussian_likelihood(mean, logs, z)

        nll = (-objective) / (math.log(2.0) * C * F * T)
        return z, nll

    def set_actnorm_init(self):
        for m in self.modules():
            if isinstance(m, ActNorm2d):
                m.inited = True
