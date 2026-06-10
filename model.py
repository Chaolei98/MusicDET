import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Union
try:
    from feature_extraction import MERT, XLSR
    from NF_model.glow_model import MultiBandGlowC
except ModuleNotFoundError:
    from .feature_extraction import MERT, XLSR
    from .NF_model.glow_model import MultiBandGlowC
import torchaudio
import random


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

        att_map = F.softmax(att_map, dim=-2)

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

        att_map = F.softmax(att_map, dim=-2)

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

        att_map = F.softmax(att_map, dim=-2)

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

class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs,
        h: graph data (#bs,
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs,
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out

class SSLAASIST(nn.Module):
    def __init__(self):
        super().__init__()

        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))
        self.LL = nn.Linear(1024, 128)

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),

        )
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

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

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        if isinstance(x, list):
            if len(x) == 1:
                x = x[0]
        x = x.squeeze(dim=1)

        x = self.LL(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)

        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)    

        w = self.attention(x)

        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

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

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden,output

class XLSRAASIST(nn.Module):
    def __init__(self, model_dir, device='cuda', freeze = True, visual=False):
        super(XLSRAASIST, self).__init__()

        self.wav2vec2 = XLSR(
            model_dir=model_dir,
            device=device,
            freeze=freeze,
            visual=visual
        )

        self.w2vaasist = SSLAASIST()
        self.visual = visual

    def extract_xlsr_features(self, audio_data):
        """
        Extract XLSR features from input waveforms.
        """
        features = self.wav2vec2.extract_features(audio_data)

        return features

    def forward(self, audio_data):

        if self.visual:
            features, attention_weights = self.wav2vec2.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
            return last_hidden, output, attention_weights
        features = self.wav2vec2.extract_features(audio_data)

        last_hidden, output = self.w2vaasist(features)
        return last_hidden, output

    def train(self, mode=True):
        if mode:
            self.w2vaasist.train(mode)
        else:
            self.w2vaasist.eval()

    def eval(self):
        self.w2vaasist.eval()
        self.wav2vec2.eval()

class MERTAASIST(nn.Module):
    def __init__(self, model_dir, device='cuda',freeze = True):
        super(MERTAASIST, self).__init__()

        self.MERT = MERT(
            model_dir=model_dir,
            device=device,
            freeze=freeze
        )

        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        features = self.MERT.extract_features(audio_data)

        last_hidden, output = self.w2vaasist(features)
        return last_hidden, output

    def train(self, mode=True):
        if mode:
            self.w2vaasist.train(mode)
        else:
            self.w2vaasist.eval()

    def eval(self):
        self.w2vaasist.eval()
        self.MERT.eval()

class SpecAugmentFT(nn.Module):
    def __init__(
        self,
        p=0.8,
        num_freq_masks=1,
        num_time_masks=1,
        time_width=(10, 20),
        highfreq_bins=(6, 20),
        fill="zero",
    ):
        super().__init__()
        self.p = p
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.time_width = time_width
        self.highfreq_bins = highfreq_bins
        self.freq_width = highfreq_bins
        self.fill = fill

    def _fill_value(self, S):
        if self.fill == "mean":
            return S.mean()
        return 0.0

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (random.random() > self.p):
            return S
        B, F, T = S.shape
        out = S.clone()
        fv = self._fill_value(out)

        for _ in range(self.num_freq_masks):
            w = random.randint(*self.freq_width)
            if w >= F:
                continue
            f0 = random.randint(0, F - w)
            out[:, f0:f0+w, :] = fv

        for _ in range(self.num_time_masks):
            w = random.randint(*self.time_width)
            if w >= T:
                continue
            t0 = random.randint(0, T - w)
            out[:, :, t0:t0+w] = fv

        return out

class SpecBandMask3D(nn.Module):
    """
    Mask one-sided spectrogram: S [B, F, T], F = n_fft//2 + 1
    mode:
      - "keep": only keep [f0,f1], others -> fill_value
      - "drop": only drop [f0,f1], others kept
    test_only:
      - True: only apply when model.eval()
      - False: apply in both train/eval
    """
    def __init__(self, sr=16000, n_fft=512, band_hz=(0.0, 8000.0),
                 mode="keep", test_only=True, fill_value=0.0):
        super().__init__()
        assert mode in ["keep", "drop"]
        self.sr = sr
        self.n_fft = n_fft
        self.band_hz = band_hz
        self.mode = mode
        self.test_only = test_only
        self.fill_value = fill_value
        self.F = n_fft // 2 + 1

    def _hz_to_bin(self, hz: float) -> int:
        k = int(round(hz * self.n_fft / self.sr))
        return max(0, min(self.F - 1, k))

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        if self.test_only and self.training:
            return S
        assert S.dim() == 3, f"Expect [B,F,T], got {S.shape}"
        f0, f1 = self.band_hz
        b0, b1 = self._hz_to_bin(f0), self._hz_to_bin(f1)
        if b1 < b0:
            b0, b1 = b1, b0

        device, dtype = S.device, S.dtype
        mask_f = torch.zeros(self.F, device=device, dtype=dtype)
        mask_f[b0:b1 + 1] = 1.0
        mask = mask_f.view(1, -1, 1)

        if self.mode == "keep":
            return S * mask + self.fill_value * (1.0 - mask)
        else:
            return S * (1.0 - mask) + self.fill_value * mask

class SpecNF(nn.Module):
    def __init__(self, K=8, L=1, R=10,
                 sr=16000,):  
        super(SpecNF, self).__init__()

        self.sr = sr
        self.n_fft = 512
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=160, win_length=512, power=2, normalized=False
        )

        self.spec_aug = SpecAugmentFT(p=0.5, fill="zero")

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.l = L
        self.eps = 1e-10

        self.mb_glow = MultiBandGlowC(
            image_shape=(32, 64, 100),
            n_bands=2,
            hidden_channels=256,
            K=K,
            L=1,
            actnorm_scale=1.0,
            flow_permutation='permute',
            flow_coupling='affine',
            LU_decomposed=True,
            learn_top=False,
            R=R,
            global_K=1,
        )

        self.first_bn = nn.BatchNorm2d(num_features=32)
        self.selu = nn.SELU(inplace=True)

        self.initialize_params()

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x, y_onehot=None):
        x = x.float()

        S = self.spec(x)

        x = torch.log(S + self.eps).unsqueeze(dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.first_bn(x)
        x = self.selu(x)

        x = x[:,:,1:,1:]

        z, nll = self.mb_glow(x, y_onehot)
        return z, nll
