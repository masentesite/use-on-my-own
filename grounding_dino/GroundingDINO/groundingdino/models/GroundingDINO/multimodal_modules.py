# ------------------------------------------------------------------------
# Grounding DINO — RGB / IR / Depth 多模态改造
# ------------------------------------------------------------------------
# 对应技术方案:
#   §6  IR 分支设计          -> IREncoder / BottleneckAdapter2d
#   §7  Depth 分支设计       -> DepthPreprocessor / DepthEncoder
#   §9  Fusion 最终定义      -> LanguageGuidedFusion (第一版, 保留作消融)
#   §10 解决分布不匹配的 Adapter -> BottleneckAdapter / BottleneckAdapter2d
#   §14 Modality Dropout     -> ModalityAugment
#
# 第二版技术方案 (V2):
#   V2 §5.1 局部 cross-attention    -> LocalCrossAttention2d
#   V2 §5.2 空间矩阵 gate           -> SpatialMixtureGate
#   V2 §5   第二版主融合            -> CrossAttentionGatedFusion
#
# 两版 Fusion 的区别(保留两套是为了 V2-E4 消融, 见 V2 §11):
#
#   第一版 (LanguageGuidedFusion) —— 辅助模态只提供「加在 RGB 上的一点残差」:
#       F_new = F_rgb + β_ir·Δ_ir + β_depth·Δ_depth
#   β 是每个 level 一个标量且零初始化, 训练后辅助支路的有效贡献很容易被压到很小。
#
#   第二版 (CrossAttentionGatedFusion) —— 辅助模态参与**生成候选特征**,
#   再由空间矩阵 gate 在每个位置决定用哪个候选:
#       A_ir    = LocalCrossAttention2d(Q=F_rgb, K/V=F_ir)
#       A_depth = LocalCrossAttention2d(Q=F_rgb, K/V=F_depth)
#       W       = SpatialMixtureGate([F_rgb, A_ir, A_depth, T_global])   # B x 3 x H x W
#       F_new   = W[:,0]·F_rgb + W[:,1]·A_ir + W[:,2]·A_depth
#
# 注意第二版**不是** zero-init: gate bias 初值 [2.0, 0.0, 0.0] 让辅助模态从第一步
# 就有非零权重入口 (§5.3)。第一版的 beta 零初始化保留在 LanguageGuidedFusion 里。
# ------------------------------------------------------------------------
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from groundingdino.util.misc import NestedTensor

from .backbone.swin_transformer import build_swin_transformer

__all__ = [
    "BottleneckAdapter",
    "BottleneckAdapter2d",
    "MultiScaleProjection",
    "DepthPreprocessor",
    "ConvNeXtBlock",
    "DepthEncoder",
    "IREncoder",
    "LanguageGuidedFusion",
    "LocalCrossAttention2d",
    "SpatialMixtureGate",
    "CrossAttentionGatedFusion",
    "ModalityAugment",
    "masked_mean_pool",
]


def masked_mean_pool(token_feats: torch.Tensor, token_mask: torch.Tensor, eps: float = 1e-6):
    """掩码平均池化, 得到句子级文本向量 (方案 §8)。

    Args:
        token_feats: [B, L, C] —— 即 text_dict["encoded_text"]
        token_mask:  [B, L]    —— 即 text_dict["text_token_mask"], True 表示真实 token

    Returns:
        [B, C]
    """
    weight = token_mask.to(token_feats.dtype).unsqueeze(-1)  # B, L, 1
    return (token_feats * weight).sum(dim=1) / (weight.sum(dim=1) + eps)


# ======================================================================
#  Adapter (方案 §10)
# ======================================================================
class BottleneckAdapter(nn.Module):
    """作用在 token 张量上的 bottleneck residual adapter。

        Adapter(x) = W_out(GELU(W_mid(LN(x))))      W_out 零初始化
        y = x + Adapter(x)

    只在最后一维上运算, 所以 encoder 的 [bs, Σhw, C] 与 decoder 的 [Nq, bs, C]
    可以共用同一个类。W_out 零初始化 => 训练起点 y == x, 模型等价于原始 RGB 模型。
    """

    def __init__(self, dim: int, bottleneck_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        """重新施加初始化。Transformer._reset_parameters() 会对所有 dim>1 的参数做
        xavier_uniform_, 会摧毁 up 层的零初始化, 所以那之后必须再调一次本方法。"""
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.down(self.norm(x)))))


class BottleneckAdapter2d(nn.Module):
    """BottleneckAdapter 的特征图版本, 用于 IR / Depth 分支的 domain adapter (§6 §9)。

    输入 [B, C, H, W], 用 1x1 Conv 代替 Linear, 空间维度逐点独立。
    同样 W_out 零初始化 => 训练起点是恒等映射。
    """

    def __init__(self, dim: int, bottleneck_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)  # 通道维 LayerNorm, 与 batch size 解耦
        self.down = nn.Conv2d(dim, bottleneck_dim, kernel_size=1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(bottleneck_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.down(self.norm(x)))))


class MultiScaleProjection(nn.Module):
    """把辅助模态的多尺度特征逐 level 投到 hidden_dim。

    结构与 GroundingDINO 自己的 input_proj (groundingdino.py 的 1x1 Conv + GroupNorm)
    保持一致, 便于逐 level 对齐到 RGB 的 srcs。
    """

    def __init__(self, in_channels: Sequence[int], hidden_dim: int = 256, num_groups: int = 32):
        super().__init__()
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, hidden_dim, kernel_size=1),
                    nn.GroupNorm(num_groups, hidden_dim),
                )
                for c in in_channels
            ]
        )
        for seq in self.proj:
            nn.init.xavier_uniform_(seq[0].weight, gain=1)
            nn.init.constant_(seq[0].bias, 0)

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [self.proj[i](f) for i, f in enumerate(feats)]


# ======================================================================
#  Depth 分支 (方案 §7)
# ======================================================================
class DepthPreprocessor(nn.Module):
    """把 16-bit 原始深度编码成 3 通道 (方案 §7):

        valid   = D > depth_min                     (0 / 小值视为 invalid)
        D_norm  = clamp((D - d_min)/(d_max - d_min), 0, 1) * valid
        M_valid = valid
        G_depth = normalize(|∇x D_norm| + |∇y D_norm|) * valid
        X_depth = concat(D_norm, M_valid, G_depth)

    归一化区间的选取很关键。实测本仓库 TrainSet. 的 depth:invalid(=0) 像素占比在
    2.3%~61.4% 之间大幅波动, 每张图有效值中位数 2500~9900、最大值 8962~19999。
    固定区间 (0, 20000) 会把只到 ~10000 的图压进 [0, 0.5], 损失一半动态范围,
    所以默认用逐样本分位数 (percentile) 模式。
    """

    def __init__(
        self,
        mode: str = "percentile",
        depth_min: float = 0.0,
        depth_max: float = 20000.0,
        low_percentile: float = 1.0,
        high_percentile: float = 99.0,
        grad_scale: float = 8.0,
        max_percentile_samples: int = 65536,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert mode in ("percentile", "fixed"), f"unknown depth_norm mode {mode!r}"
        self.mode = mode
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        # |gx| + |gy| 在 D_norm ∈ [0,1] 上的理论最大值是 8 (Sobel 正权重之和 4+4),
        # 用它做固定尺度归一化:确定性、可解释, 且不需要额外的分位数统计。
        self.grad_scale = grad_scale
        self.max_percentile_samples = max_percentile_samples
        self.eps = eps

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t().contiguous()
        # persistent=False: 不进 state_dict, 避免给 checkpoint 添加无关键
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _percentile_range(self, depth: torch.Tensor, valid: torch.Tensor):
        """逐样本用有效像素的分位数估计 (d_min, d_max)。

        对有效像素等步长下采样后再取分位数:1080p 单图有 ~1M 个有效像素,
        全量 torch.quantile 在 CPU 上要几十毫秒, 下采样到 64k 后统计上完全够用。
        """
        b = depth.shape[0]
        d_min = depth.new_zeros(b)
        d_max = depth.new_ones(b)
        for i in range(b):
            # depth[i] 是 [1,H,W], valid[i] 是 [H,W], 都拉平后再做掩码选取
            v = depth[i].reshape(-1)[valid[i].reshape(-1)]
            if v.numel() == 0:
                continue
            step = max(1, v.numel() // self.max_percentile_samples)
            v = v[::step].float().contiguous()
            d_min[i] = torch.quantile(v, self.low_percentile / 100.0)
            d_max[i] = torch.quantile(v, self.high_percentile / 100.0)
        # 平坦深度会让 d_max == d_min, 加 eps 兜住除零
        return d_min, torch.maximum(d_max, d_min + self.eps)

    def forward(self, depth: torch.Tensor, valid: Optional[torch.Tensor] = None):
        """Args:
            depth: [B, 1, H, W] 或 [B, H, W], 原始深度值 (float)
            valid: [B] 或 [B,1,H,W] bool, 外部给定的有效性 (如深度整路缺失)。None 视为全有效。

        Returns:
            x_depth: [B, 3, H, W]
            valid_map: [B, 1, H, W] bool
        """
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        assert depth.dim() == 4 and depth.shape[1] == 1, (
            f"depth 期望 [B,1,H,W], 得到 {tuple(depth.shape)}"
        )
        depth = depth.float()
        b = depth.shape[0]

        # 1) 有效性:0 / 小值视为 invalid
        valid_map = depth > self.depth_min
        if self.mode == "fixed":
            valid_map = valid_map & (depth < self.depth_max)
        if valid is not None:
            v = valid if valid.dim() == 4 else valid.view(b, 1, 1, 1)
            valid_map = valid_map & v.to(valid_map.device).bool()

        # 2) 归一化深度
        if self.mode == "percentile":
            d_min, d_max = self._percentile_range(depth, valid_map[:, 0])
        else:
            d_min = depth.new_full((b,), self.depth_min)
            d_max = depth.new_full((b,), self.depth_max)
        scale = (d_max - d_min).view(b, 1, 1, 1)
        d_norm = ((depth - d_min.view(b, 1, 1, 1)) / scale).clamp(0.0, 1.0)
        d_norm = d_norm * valid_map.to(d_norm.dtype)

        # 3) 局部深度边缘
        gx = F.conv2d(d_norm, self.sobel_x.to(d_norm.dtype), padding=1)
        gy = F.conv2d(d_norm, self.sobel_y.to(d_norm.dtype), padding=1)
        grad = ((gx.abs() + gy.abs()) / self.grad_scale).clamp(0.0, 1.0)
        grad = grad * valid_map.to(grad.dtype)

        x_depth = torch.cat([d_norm, valid_map.to(d_norm.dtype), grad], dim=1)
        return x_depth, valid_map


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 风格残差块: DWConv7x7 -> Norm -> 1x1 升维 -> GELU -> 1x1 降维。

    用 GroupNorm(1, C) 而不是 BatchNorm:深度分支的 batch 统计量在 batch size 小时
    不可靠, 而且 GroupNorm 在 train/eval 下行为一致, 便于冻结 / 复现。
    layer-scale gamma 初始 1e-6, 保证堆叠时训练初期接近恒等。
    """

    def __init__(self, dim: int, mlp_ratio: int = 4, layer_scale_init: float = 1e-6):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + self.gamma.view(1, -1, 1, 1) * h


class DepthEncoder(nn.Module):
    """轻量 CNN / ConvNeXt 风格金字塔 (方案 §7)。

    深度分支的目标不是学 RGB 纹理, 而是学距离、几何边界和空间布局, 所以不强行复用 RGB Swin。

    下采样全部用 k=3, s=2, p=1, 逐级得到 ceil(H/2)、ceil(H/4)、ceil(H/8)、ceil(H/16)、ceil(H/32)。
    以 750x1333 的输入为例, 三个输出尺度是 94x167、47x84、24x42, 与 RGB Swin 的
    level 0/1/2 逐像素对齐(实测吻合, 见 test_multimodal.py 的 shape test)。
    """

    def __init__(
        self,
        in_chans: int = 3,
        widths: Sequence[int] = (32, 64, 128, 256, 512),
        depths: Sequence[int] = (0, 2, 2, 2, 2),
        mlp_ratio: int = 4,
    ):
        super().__init__()
        assert len(widths) == 5 and len(depths) == 5
        w0, w1, w2, w3, w4 = widths

        def stage(cin, cout, n_blocks):
            layers: List[nn.Module] = [
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(1, cout),
                nn.GELU(),
            ]
            layers += [ConvNeXtBlock(cout, mlp_ratio=mlp_ratio) for _ in range(n_blocks)]
            return nn.Sequential(*layers)

        self.stem = stage(in_chans, w0, depths[0])  # H/2
        self.stage1 = stage(w0, w1, depths[1])  # H/4
        self.stage2 = stage(w1, w2, depths[2])  # H/8   -> out[0]
        self.stage3 = stage(w2, w3, depths[3])  # H/16  -> out[1]
        self.stage4 = stage(w3, w4, depths[4])  # H/32  -> out[2]

        self.out_channels = [w2, w3, w4]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        o0 = self.stage2(x)
        o1 = self.stage3(o0)
        o2 = self.stage4(o1)
        return [o0, o1, o2]


# ======================================================================
#  IR 分支 (方案 §6 方案 A)
# ======================================================================
class IREncoder(nn.Module):
    """IR 分支:1 通道热图 -> Conv2d(1,3,1) stem -> 独立 Swin-T。

    方案 A 的推荐做法是「IR 先变成 1 通道, 用 1->3 stem 再接独立 IR Swin,
    第一层权重由 RGB 第一层在输入通道维度求均值初始化」。

    这里把 stem 权重设为 1/3, 于是
        Swin.patch_embed.proj ∘ stem  ==  1/3 * Σ_c proj[:, c, :, :]
    正好是「patch_embed 权重沿输入通道求均值」。因为 Swin 侧仍是标准 in_chans=3,
    RGB Swin 的 state_dict 可以逐键直接拷过来(已实测:187 个键、形状全一致),
    warm-start 就是一次 load_state_dict, 不需要任何权重改形。

    warm-start 由 GroundingDINO.load_state_dict 的 post hook 触发, 见 groundingdino.py。
    """

    def __init__(
        self,
        modelname: str = "swin_T_224_1k",
        pretrain_img_size: int = 224,
        out_indices: Sequence[int] = (1, 2, 3),
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.stem = nn.Conv2d(1, 3, kernel_size=1)
        nn.init.constant_(self.stem.weight, 1.0 / 3.0)
        nn.init.zeros_(self.stem.bias)

        self.body = build_swin_transformer(
            modelname,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(out_indices),
            dilation=False,
            use_checkpoint=use_checkpoint,
        )
        self.out_indices = tuple(out_indices)
        self.out_channels = [self.body.num_features[i] for i in self.out_indices]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Args:
            x:    [B, 1, H, W] 或 [B, 3, H, W](3 通道时先取通道均值统一成 1 通道热图)
            mask: [B, H, W] bool padding mask, 仅用于满足 Swin.forward 的 NestedTensor 接口,
                  返回的 mask 会被丢弃 —— 融合侧一律复用 RGB 的 mask。
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        elif x.shape[1] != 1:
            raise ValueError(f"IR 输入通道数应为 1 或 3, 得到 {x.shape[1]}")

        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[-2], x.shape[-1], dtype=torch.bool, device=x.device)
        out = self.body(NestedTensor(self.stem(x), mask))
        return [out[i].tensors for i in range(len(out))]


# ======================================================================
#  语言引导的残差融合 (方案 §9)
# ======================================================================
class LanguageGuidedFusion(nn.Module):
    """逐 level 的「RGB 主路径 + IR/Depth 残差注入」(方案 §9)。

        T_global = masked_mean_pool(encoded_text, text_token_mask)
        F_ir,l   = ir_adapter_l(ir_proj_l(ir_feat_l))
        G_ir,l   = sigmoid(gate_l([F_rgb,l, F_ir,l, T_global 广播]))
        ΔF_ir,l  = delta_l(G_ir,l * F_ir,l)
        F_new,l  = F_rgb,l + valid_ir * β_ir,l * ΔF_ir,l
                          + valid_d  * β_d,l  * ΔF_d,l

    梯度能流动的两个必要条件(否则辅助支路永远学不动):

    1. **β 零初始化是安全的**:β=0 时 ΔF 拿不到梯度(∂/∂ΔF = β = 0), 但 β 自己的梯度
       ∂L/∂β = ⟨∂L/∂F_new, ΔF⟩ ≠ 0, 所以优化器会先把 β 推离 0, ΔF 随后开始学习。
    2. **delta 卷积绝不能零初始化**:若 ΔF ≡ 0, 则 ∂L/∂β = 0, β 也永远不动,
       整条支路彻底死掉。零初始化只加在 Adapter 的输出层(Adapter 是残差块, 零初始化=恒等)。

    文本侧只读 text_dict["encoded_text"], 不重复跑 BERT;token 级的图文交互仍由原有的
    Feature Enhancer 与 Decoder 完成。
    """

    def __init__(
        self,
        dim: int = 256,
        num_levels: int = 3,
        text_dim: int = 256,
        adapter_dim: int = 64,
        gate_bias: float = -2.0,
        beta_init: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        self.text_proj = nn.Linear(text_dim, dim)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

        def make_gates():
            return nn.ModuleList([nn.Conv2d(dim * 3, dim, kernel_size=1) for _ in range(num_levels)])

        def make_deltas():
            return nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=1) for _ in range(num_levels)])

        self.ir_adapters = nn.ModuleList(
            [BottleneckAdapter2d(dim, adapter_dim) for _ in range(num_levels)]
        )
        self.depth_adapters = nn.ModuleList(
            [BottleneckAdapter2d(dim, adapter_dim) for _ in range(num_levels)]
        )
        self.ir_gates = make_gates()
        self.depth_gates = make_gates()
        self.ir_deltas = make_deltas()
        self.depth_deltas = make_deltas()

        # gate 权重用小方差 + 负 bias:训练初期 sigmoid ≈ 0.12, 门控不会一上来就全开
        for gate in list(self.ir_gates) + list(self.depth_gates):
            nn.init.normal_(gate.weight, std=0.02)
            nn.init.constant_(gate.bias, gate_bias)
        for delta in list(self.ir_deltas) + list(self.depth_deltas):
            nn.init.xavier_uniform_(delta.weight)
            nn.init.zeros_(delta.bias)

        # 每个 level 一个可学习标量 (方案 §9:第一版用标量, 初始化为 0)
        self.beta_ir = nn.Parameter(torch.full((num_levels,), float(beta_init)))
        self.beta_depth = nn.Parameter(torch.full((num_levels,), float(beta_init)))

        # 最近一次 forward 的可观测指标。第一版没有空间 gate, 能观测的就是 beta:
        # 它就是「辅助模态贡献被压缩成多大」的直接读数。保持与
        # CrossAttentionGatedFusion.last_stats 同样的「detach 后的标量张量字典」约定,
        # 让训练脚本可以用同一段代码打印两版融合(V2 §9 的两版对照)。
        self.last_stats = None

    def _branch(self, rgb, aux, text_up, adapter, gate, delta, beta):
        """计算单个模态在单个 level 上的 ΔF。"""
        if aux.shape[-2:] != rgb.shape[-2:]:
            # 兜住奇数尺寸下 ceil 取整的偏差, 兑现「融合输出与 RGB srcs 完全同形状」
            aux = F.interpolate(aux, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        aux = adapter(aux)

        b, c, h, w = rgb.shape
        text_b = text_up.view(b, c, 1, 1).expand(-1, -1, h, w)
        g = torch.sigmoid(gate(torch.cat([rgb, aux, text_b], dim=1)))
        return beta * delta(g * aux)

    def forward(
        self,
        rgb_srcs: Sequence[torch.Tensor],
        ir_srcs: Optional[Sequence[torch.Tensor]] = None,
        depth_srcs: Optional[Sequence[torch.Tensor]] = None,
        text_dict: Optional[dict] = None,
        ir_valid: Optional[torch.Tensor] = None,
        depth_valid: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            rgb_srcs:   长度 num_levels 的 [B, 256, h, w] 列表
            ir_srcs:    同形状列表, 或 None(IR 缺失)
            depth_srcs: 同形状列表, 或 None(Depth 缺失)
            text_dict:  含 encoded_text [B,L,256] 与 text_token_mask [B,L]
            ir_valid / depth_valid: [B] bool, 整路模态是否有效; None 视为全有效

        Returns:
            长度 num_levels 的融合特征列表, 每个元素与对应 rgb_srcs 同形状
        """
        if ir_srcs is None and depth_srcs is None:
            self.last_stats = None
            return list(rgb_srcs)
        assert text_dict is not None, "启用 Fusion 时必须传入 text_dict(用于语言引导的门控)"
        assert len(rgb_srcs) == self.num_levels, (
            f"rgb_srcs 应有 {self.num_levels} 个 level, 得到 {len(rgb_srcs)}"
        )
        if ir_srcs is not None:
            assert len(ir_srcs) == self.num_levels, (
                f"ir_srcs 应有 {self.num_levels} 个 level, 得到 {len(ir_srcs)}"
            )
        if depth_srcs is not None:
            assert len(depth_srcs) == self.num_levels, (
                f"depth_srcs 应有 {self.num_levels} 个 level, 得到 {len(depth_srcs)}"
            )

        t_global = masked_mean_pool(text_dict["encoded_text"], text_dict["text_token_mask"])
        text_up = self.text_proj(t_global)  # B x dim

        b = rgb_srcs[0].shape[0]
        device = rgb_srcs[0].device
        # valid flag 必须真的参与运算:否则「整路置零」会被误读成真实的黑色 / 零距离信息
        # _as_valid 内部已保证按样本折叠成 [B]
        ir_valid = self._as_valid(ir_valid, b, device)
        depth_valid = self._as_valid(depth_valid, b, device)
        out = []
        for lvl, rgb in enumerate(rgb_srcs):
            fused = rgb
            if ir_srcs is not None:
                fused = fused + self._branch(
                    rgb, ir_srcs[lvl], text_up, self.ir_adapters[lvl],
                    self.ir_gates[lvl], self.ir_deltas[lvl], self.beta_ir[lvl],
                ) * ir_valid.view(b, 1, 1, 1).to(fused.dtype)
            if depth_srcs is not None:
                fused = fused + self._branch(
                    rgb, depth_srcs[lvl], text_up, self.depth_adapters[lvl],
                    self.depth_gates[lvl], self.depth_deltas[lvl], self.beta_depth[lvl],
                ) * depth_valid.view(b, 1, 1, 1).to(fused.dtype)
            out.append(fused)

        self.last_stats = {}
        for l in range(self.num_levels):
            self.last_stats[f"beta_ir/l{l}"] = self.beta_ir[l].detach()
            self.last_stats[f"beta_depth/l{l}"] = self.beta_depth[l].detach()
        return out

    @staticmethod
    def _as_valid(valid, b: int, device) -> torch.Tensor:
        """把 valid 统一成 [B] 的逐样本 bool。

        接受三种输入:
          - None:           视为「该模态整路有效」(语义上不同于「该模态缺失」)
          - [B]:            逐样本的有效性
          - [B,1,H,W]/[B,H,W]: 逐像素的有效性, 按样本折叠成 [B]

        ⚠️ 必须显式处理逐像素的情形。`ModalityAugment._merge_valid` 返回的是
        `valid & keep`, 形状与输入一致; `DepthPreprocessor` 的 docstring 也明确接受
        `[B,1,H,W]`。如果这里直接 `.view(b,1,1,1)`, 逐像素 mask 会以
        "shape '[2,1,1,1]' is invalid for input of size 32" 崩掉。
        """
        if valid is None:
            return torch.ones(b, dtype=torch.bool, device=device)
        valid = valid.to(device).bool()
        if valid.dim() == 1:
            return valid
        assert valid.shape[0] == b, (
            f"valid 的 batch 维 {valid.shape[0]} 与特征的 batch {b} 不一致"
        )
        # [B, ...] -> [B], 逐像素全有效才算该样本有效
        return valid.reshape(b, -1).all(dim=-1)


# ======================================================================
#  第二版: 局部 Cross-Attention + 空间矩阵 Gate (V2 方案 §5)
# ======================================================================
class LocalCrossAttention2d(nn.Module):
    """RGB 在 k x k 局部窗口内查询辅助模态 (V2 §5.1)。

        q = q_proj(norm_q(rgb));  k = k_proj(norm_kv(aux));  v = v_proj(norm_kv(aux))
        A = softmax(q·kᵀ / √d) · v

    输入输出都是 [B, dim, H, W], 逐 level 独立。

    **为什么不用 F.unfold 实现局部窗口。** 方案 §4 要求局部窗口而不是全局
    H*W × H*W attention, 但 unfold 本身同样会炸显存: H/8 level(750×1333 → 94×167
    ≈ 15700 token)、dim=256、window=5 时, 单个 unfold 输出就是
    B × (256×25) × 15700 —— batch=4 约 1.6 GB/k 和 v **各一份**, 反向还要再留一份,
    三个 level 两个模态叠起来超过 10 GB, 正好命中方案 §12 第一行「显存突然暴涨」。

    所以这里改成「按窗口偏移量循环」: 每个偏移只把 k/v 平移一次(pad + 切片, 零拷贝),
    用逐元素乘加累加 logits 与输出。结果与 unfold 版**逐位等价**, 中间张量却只有
    O(B·C·H·W) 的临时量和 O(B·heads·H·W·k²) 的 attn 矩阵 —— 与 window_size 成
    线性, 而不是与通道数相乘。

    越界位置用 0 填充而不是 -inf: k=0 ⇒ logit=0, 该偏移只是「和中心同量级的一个
    中性候选」; 用 -inf 则在整窗口都在图外的极端像素上会得到全 -inf → softmax NaN。
    """

    def __init__(self, dim: int = 256, num_heads: int = 8, window_size: int = 5,
                 dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} 不能被 num_heads={num_heads} 整除"
        assert window_size >= 1 and window_size % 2 == 1, "window_size 必须是正奇数"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # GroupNorm(1, dim) 而不是 BatchNorm: 与 BottleneckAdapter2d 同样的理由 ——
        # batch 统计量在小 batch / 冻结-评估态下不可靠, GroupNorm 在 train/eval 下一致。
        self.norm_q = nn.GroupNorm(1, dim)
        self.norm_kv = nn.GroupNorm(1, dim)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        """xavier 初始化四个 1x1 投影。

        ⚠️ **刻意不做零初始化**。方案 §5.4 明确要求候选特征不要零初始化: 若
        A_aux ≡ 0, 而 gate 又把 W[:,1]/W[:,2] 分给这个全零候选(V2 默认初值是
        0.11), F_new 就会在训练起点损失掉 22% 的幅值 —— 相当于悄悄缩放了输入给
        Transformer 的特征。xavier 让 out_proj 的输出方差与输入同量级(1x1 conv,
        fan_in = dim), F_new 因此始终是一个良态的凸组合。
        """
        for conv in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

    def _window_attention(self, q, k, v):
        """q/k/v: [B, heads, head_dim, H, W] -> [B, heads, head_dim, H, W]。"""
        p = self.window_size // 2
        b, heads, head_dim, h, w = q.shape
        dtype = q.dtype
        kp = F.pad(k, (p, p, p, p))
        vp = F.pad(v, (p, p, p, p))

        offsets = [(dy, dx)
                   for dy in range(self.window_size)
                   for dx in range(self.window_size)]

        # ---- ① logits: q 与窗口内每个偏移的 k 做点积 ----
        # 在 fp32 里累加: head_dim 维点积在 fp16 下精度只有 ~3 位十进制, softmax
        # 对 logits 的绝对误差很敏感。每步只是一次 [B,heads,d,H,W] 的临时量。
        logits = [
            (q.float() * kp[..., dy:dy + h, dx:dx + w].float()).sum(dim=2)
            for dy, dx in offsets
        ]
        logits = torch.stack(logits, dim=-1) * self.scale  # [B,heads,H,W,k*k]
        attn = self.dropout(torch.softmax(logits, dim=-1))  # fp32, 最后维归一

        # ---- ② 用同一组权重对平移后的 v 加权求和 ----
        out = None
        for i, (dy, dx) in enumerate(offsets):
            term = attn[..., i].unsqueeze(2) * vp[..., dy:dy + h, dx:dx + w].float()
            out = term if out is None else out + term
        return out.to(dtype)

    def forward(self, rgb: torch.Tensor, aux: torch.Tensor,
                aux_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            rgb: [B, C, H, W] —— 提供 query
            aux: [B, C, H', W'] —— 提供 key/value, 尺寸不一致时 bilinear 对齐到 rgb
            aux_valid: [B] 或 [B,1,H,W] bool。**本模块不用它做逐像素屏蔽**(遮蔽由
                SpatialMixtureGate 负责, 方案 §5.2), 只在「整路都无效」时直接返回
                零特征, 省掉一次注定被 mask 掉的 attention 计算。

        Returns:
            aux_attended: [B, C, H, W], 与 rgb 同形状
        """
        if aux_valid is not None and not bool(aux_valid.to(rgb.device).bool().any()):
            return torch.zeros_like(rgb)

        if aux.shape[-2:] != rgb.shape[-2:]:
            # 兜住奇数尺寸下 ceil 取整的偏差, 兑现「候选特征与 RGB 同形状」
            aux = F.interpolate(aux, size=rgb.shape[-2:], mode="bilinear",
                                align_corners=False)

        b, c, h, w = rgb.shape
        q = self.q_proj(self.norm_q(rgb)).view(b, self.num_heads, self.head_dim, h, w)
        kv = self.norm_kv(aux)
        k = self.k_proj(kv).view(b, self.num_heads, self.head_dim, h, w)
        v = self.v_proj(kv).view(b, self.num_heads, self.head_dim, h, w)

        out = self._window_attention(q, k, v)
        return self.out_proj(out.reshape(b, c, h, w))


class SpatialMixtureGate(nn.Module):
    """输出 B x 3 x H x W 的 softmax 空间权重 (V2 §5.2)。

        gate_logits = conv([F_rgb, A_ir, A_depth, T_global_up])   # B x 3 x H x W
        缺失模态: gate_logits[:, k] = -1e4
        W = softmax(gate_logits, dim=1)

    通道 0 / 1 / 2 分别对应 RGB / IR-attended / Depth-attended。

    「缺失模态把 logit 置 -1e4」而不是「把输入置零」是方案 §13 的第 7 条硬要求:
    置零只说明「这个模态的特征是 0」, 模型仍然可以给它分权重; 置 -1e4 才是
    「这个模态这一路根本不可用」。两者在 RGB+IR / RGB+Depth / RGB+IR+Depth 三种
    输入组合下共用同一套三通道 gate, 不需要任何分支。
    """

    def __init__(self, dim: int = 256, num_levels: int = 3, text_dim: int = 256,
                 rgb_bias: float = 2.0, aux_bias: float = 0.0, use_text: bool = True,
                 init_std: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.use_text = use_text
        self.rgb_bias = float(rgb_bias)
        self.aux_bias = float(aux_bias)

        if use_text:
            self.text_proj = nn.Linear(text_dim, dim)
            nn.init.xavier_uniform_(self.text_proj.weight)
            nn.init.zeros_(self.text_proj.bias)
        else:
            self.text_proj = None

        in_ch = dim * (4 if use_text else 3)
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_ch, 3, kernel_size=1) for _ in range(num_levels)]
        )
        for c in self.conv:
            # bias 决定初始 gate 分布: softmax([2,0,0]) = [0.786, 0.107, 0.107],
            # 与方案 §5.3 的推荐默认 [2.0, 0.0, 0.0] 一致; 单辅助模态时
            # softmax([2,-1e4,0]) = [0.881, 0, 0.119], 也符合 §5.3 的 0.88/0.12。
            #
            # 权重用小方差而**不是全零**: 方案 §5.3 要求「gate 不要初始化为全 0」。
            # std=1e-3 时 logits 的随机扰动只有 ~0.03(1024 个输入通道,
            # 1e-3·√1024·O(1)), 初始分布基本就是 bias 决定的 0.786/0.107/0.107,
            # 满足 §10 的 gate init test; 同时打破了权重矩阵的完全对称。
            nn.init.normal_(c.weight, std=init_std)
            with torch.no_grad():
                c.bias.copy_(torch.tensor([self.rgb_bias] + [self.aux_bias] * 2))

    def forward(self, lvl: int, rgb: torch.Tensor, a_ir: Optional[torch.Tensor],
                a_depth: Optional[torch.Tensor], text_up: Optional[torch.Tensor],
                ir_ok: Optional[torch.Tensor] = None,
                depth_ok: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            lvl: level 下标
            rgb / a_ir / a_depth: [B, C, H, W]; 缺失模态传 None(内部填零)
            text_up: [B, C] 句子级文本向量, use_text=False 时忽略
            ir_ok / depth_ok: bool, 支持 [B] / [B,1] / [B,H,W] / [B,1,H,W] 四种写法,
                True 表示该位置可用。None 表示「未提供 → 全部可用」。
                ⚠️ [B] 是最常见的写法(方案 §14 的 modality dropout 就是逐样本地给出
                valid), 而 [B] **不能**直接广播到 [B,1,H,W], 所以这里统一归一化,
                而不是把形状约束推给调用方 —— 传错了只会得到一句 broadcast 报错,
                排查成本远高于这里多写几行。

        Returns:
            W: [B, 3, H, W] fp32, 沿 dim=1 归一化
        """
        b, _, h, w = rgb.shape

        # ---- 「候选不存在」本身就是缺失, 由本模块负责 mask ----
        # 方案 §13 第 7 条(缺失 ⇒ logit 置 -1e4)只有一条规则, 但「缺失」有两种来源:
        # 候选根本没传(a_ir is None)与 valid 标了 False。让两者都在这里收口, 而不是
        # 一部分留在这里、一部分推给调用方 —— 后者正是「整路缺失却忘了 mask」这个
        # 坑的成因: gate 会给那个全零候选留下 ~0.107 的权重(§5.3 的 bias 初值),
        # F_new 于是白丢 11% 的幅值(§5.4 警告的情形)。规则只有一处落点, 就不可能漏。
        parts = [rgb]
        parts.append(a_ir if a_ir is not None else torch.zeros_like(rgb))
        parts.append(a_depth if a_depth is not None else torch.zeros_like(rgb))
        if self.use_text:
            assert text_up is not None, "fusion_gate_use_text=True 时必须传入 text_up"
            parts.append(text_up.view(b, self.dim, 1, 1).expand(-1, -1, h, w))

        logits = self.conv[lvl](torch.cat(parts, dim=1)).float()  # [B,3,H,W]

        keep = torch.ones(b, 3, h, w, dtype=torch.bool, device=logits.device)
        any_mask = False
        if a_ir is None:
            keep[:, 1:2] = False
            any_mask = True
        elif ir_ok is not None:
            keep[:, 1:2] = self._as_ok(ir_ok, b)
            any_mask = True
        if a_depth is None:
            keep[:, 2:3] = False
            any_mask = True
        elif depth_ok is not None:
            keep[:, 2:3] = self._as_ok(depth_ok, b)
            any_mask = True

        if any_mask:
            # -1e4 在 fp32 下 exp() 严格下溢到 0, 该通道权重精确为 0;
            # 剩下的通道由 softmax 重新归一化, 权重不会凭空丢失
            logits = logits.masked_fill(~keep, -1e4)

        return torch.softmax(logits, dim=1)

    @staticmethod
    def _as_ok(ok: torch.Tensor, b: int) -> torch.Tensor:
        """把 [B] / [B,1] / [B,H,W] 的 valid 归一成可广播到 [B,1,H,W] 的形状。"""
        v = ok.bool()
        if v.dim() == 1:
            return v.view(b, 1, 1, 1)
        if v.dim() == 2:
            return v.view(b, 1, 1, 1)
        if v.dim() == 3:
            return v.unsqueeze(1)
        return v


class CrossAttentionGatedFusion(nn.Module):
    """第二版主融合:局部 cross-attention 候选 + 空间矩阵 gate (V2 §5)。

        A_ir,l    = candidate_norm(ir_adapter_l(LocalCrossAttention2d(F_rgb,l, F_ir,l)))
        A_depth,l = candidate_norm(depth_adapter_l(LocalCrossAttention2d(F_rgb,l, F_depth,l)))
        W_l       = SpatialMixtureGate([F_rgb,l, A_ir,l, A_depth,l, T_global])
        F_new,l   = W[:,0]·F_rgb,l + W[:,1]·A_ir,l + W[:,2]·A_depth,l

    三个候选的定位很关键(方案 §15): F_rgb 是「原始 RGB」, A_ir / A_depth 是
    「被辅助模态提示过的 RGB」, 而不是残差增量。gate 做的是在每个空间位置
    在这三个候选之间做**凸组合**(权重非负且和为 1), 所以既不会像第一版的 beta
    那样被压到接近 0, 也不会像「用辅助模态替换 RGB」那样有性能崩塌的风险。

    单模态缺失时不需要分支: 候选填零 + 对应 gate logit 置 -1e4, 权重精确为 0,
    RGB+IR / RGB+Depth / RGB+IR+Depth 三种组合走的是同一段代码。
    """

    def __init__(
        self,
        dim: int = 256,
        num_levels: int = 3,
        text_dim: int = 256,
        adapter_dim: int = 64,
        num_heads: int = 8,
        window_size: int = 5,
        gate_type: str = "spatial_softmax",
        gate_rgb_bias: float = 2.0,
        gate_aux_bias: float = 0.0,
        gate_use_text: bool = True,
        dropout: float = 0.0,
        log_stats: bool = True,
    ):
        super().__init__()
        assert gate_type == "spatial_softmax", (
            f"第二版只实现了 spatial_softmax 空间矩阵 gate, 得到 {gate_type!r}"
        )
        self.dim = dim
        self.num_levels = num_levels
        self.gate_type = gate_type
        self.log_stats = bool(log_stats)

        self.ir_attn = nn.ModuleList([
            LocalCrossAttention2d(dim, num_heads, window_size, dropout)
            for _ in range(num_levels)
        ])
        self.depth_attn = nn.ModuleList([
            LocalCrossAttention2d(dim, num_heads, window_size, dropout)
            for _ in range(num_levels)
        ])
        # ⚠️ BottleneckAdapter2d 的**零初始化加在 up 层**(残差块, 零初始化 = 恒等),
        # 不是把整个候选置零 —— 方案 §5.4「候选 adapter 不要零初始化」指的正是
        # 「不要让 A_aux ≡ 0」。这里的语义是 A_aux = cross_attn 的输出, 训练起点
        # 候选已经是一个真实特征, 而不是需要靠 beta 从 0 推起来的残差。
        self.ir_adapters = nn.ModuleList(
            [BottleneckAdapter2d(dim, adapter_dim) for _ in range(num_levels)]
        )
        self.depth_adapters = nn.ModuleList(
            [BottleneckAdapter2d(dim, adapter_dim) for _ in range(num_levels)]
        )
        # candidate_norm: 把候选标准化到与 F_rgb 可比的尺度(方案 §5.4 的
        # candidate_norm)。input_proj 用的是 GroupNorm(32, 256), 输出近似零均值
        # 单位方差; 候选若尺度差一个量级, 凸组合就等价于偷偷缩小/放大主干特征。
        self.ir_norms = nn.ModuleList([nn.GroupNorm(1, dim) for _ in range(num_levels)])
        self.depth_norms = nn.ModuleList([nn.GroupNorm(1, dim) for _ in range(num_levels)])

        self.gate = SpatialMixtureGate(
            dim=dim, num_levels=num_levels, text_dim=text_dim,
            rgb_bias=gate_rgb_bias, aux_bias=gate_aux_bias, use_text=gate_use_text,
        )

        self.last_stats = None

    # ------------------------------------------------------------------
    def _candidate(self, rgb, aux, attn, adapter, norm):
        a = attn(rgb, aux)
        return norm(adapter(a))

    @staticmethod
    def _valid_map(valid: Optional[torch.Tensor], b: int, h: int, w: int, device):
        """把任意形状的 valid 归一成可广播到 [B,1,H,W] 的 bool。

        与 LanguageGuidedFusion._as_valid 的差别: 那边必须把 [B,1,H,W] 折叠成 [B]
        (它只有标量 beta), 这里 gate 是**空间矩阵**, 所以保留逐像素信息 ——
        depth hole 这类局部缺失才能在对应位置上真正把 Depth 的权重压到 0。
        形状对不上时用 nearest 下采样到当前 level 的分辨率。
        """
        if valid is None:
            return None
        v = valid.to(device).bool()
        if v.dim() == 1:
            return v.view(b, 1, 1, 1)
        assert v.shape[0] == b, (
            f"valid 的 batch 维 {v.shape[0]} 与特征的 batch {b} 不一致"
        )
        if v.dim() == 3:
            v = v.unsqueeze(1)
        if v.shape[1] != 1:
            v = v.all(dim=1, keepdim=True)
        if v.shape[-2:] != (h, w):
            v = F.interpolate(v.float(), size=(h, w), mode="nearest") > 0.5
        return v

    @staticmethod
    def _masked_mean(x: torch.Tensor, ok: Optional[torch.Tensor]) -> torch.Tensor:
        """按 ok 掩码求均值; 整批都不 ok 时返回 0。

        为什么不是 W[:,1].mean() 的朴素平均: 开了 modality dropout 之后同一批里
        总有一部分样本的 IR 是被丢掉的, 那些样本的 W_ir 恒等于 0。朴素平均会把
        这个 0 混进来, 让 gate_ir_mean 系统性偏低 —— 而 §9 的判据恰恰是
        「gate_ir_mean 长期接近 0 表示 IR 没被使用」, 口径偏了就会误判。
        """
        if ok is None:
            return x.mean()
        m = ok.expand_as(x).to(x.dtype)
        return (x * m).sum() / m.sum().clamp_min(1.0)

    def forward(
        self,
        rgb_srcs: Sequence[torch.Tensor],
        ir_srcs: Optional[Sequence[torch.Tensor]] = None,
        depth_srcs: Optional[Sequence[torch.Tensor]] = None,
        text_dict: Optional[dict] = None,
        ir_valid: Optional[torch.Tensor] = None,
        depth_valid: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            rgb_srcs:   长度 num_levels 的 [B, 256, h, w] 列表
            ir_srcs:    同形状列表, 或 None(IR 缺失)
            depth_srcs: 同形状列表, 或 None(Depth 缺失)
            text_dict:  含 encoded_text [B,L,256] 与 text_token_mask [B,L]
            ir_valid / depth_valid: [B] 或 [B,1,H,W] bool; None 视为全有效

        Returns:
            长度 num_levels 的融合特征列表, 每个元素与对应 rgb_srcs 同形状。
            同时把 gate 统计写入 self.last_stats(detach 后的标量张量字典)。
        """
        if ir_srcs is None and depth_srcs is None:
            # 纯 RGB 路径: 与上游 GroundingDINO 逐位一致(V2 §10 regression test)
            self.last_stats = None
            return list(rgb_srcs)

        assert len(rgb_srcs) == self.num_levels, (
            f"rgb_srcs 应有 {self.num_levels} 个 level, 得到 {len(rgb_srcs)}"
        )
        if ir_srcs is not None:
            assert len(ir_srcs) == self.num_levels, (
                f"ir_srcs 应有 {self.num_levels} 个 level, 得到 {len(ir_srcs)}"
            )
        if depth_srcs is not None:
            assert len(depth_srcs) == self.num_levels, (
                f"depth_srcs 应有 {self.num_levels} 个 level, 得到 {len(depth_srcs)}"
            )

        text_up = None
        if self.gate.use_text:
            assert text_dict is not None, (
                "fusion_gate_use_text=True 时必须传入 text_dict(用于 gate 的语言条件)"
            )
            t_global = masked_mean_pool(
                text_dict["encoded_text"], text_dict["text_token_mask"]
            )
            text_up = self.gate.text_proj(t_global)  # B x dim

        b = rgb_srcs[0].shape[0]
        device = rgb_srcs[0].device

        out: List[torch.Tensor] = []
        stats: dict = {}
        for lvl, rgb in enumerate(rgb_srcs):
            _, c, h, w = rgb.shape
            a_ir = a_depth = None
            if ir_srcs is not None:
                a_ir = self._candidate(rgb, ir_srcs[lvl], self.ir_attn[lvl],
                                       self.ir_adapters[lvl], self.ir_norms[lvl])
            if depth_srcs is not None:
                a_depth = self._candidate(rgb, depth_srcs[lvl], self.depth_attn[lvl],
                                          self.depth_adapters[lvl], self.depth_norms[lvl])

            # 缺失模态的 mask 由 SpatialMixtureGate 统一负责: 候选为 None(整路缺失)
            # 与 valid 标 False(逐样本 / 逐像素缺失)在那里收口成同一条 -1e4 规则,
            # 见该模块的说明。这里只负责把 valid 折成 / 下采样成 [B,1,h,w]。
            # 注意**不做** `bool(ir_ok.any())` 这类判断 —— 那会在每个 level 每个模态
            # 上引入一次 device 同步, 训练时是纯粹的流水线停顿。
            ir_ok = (self._valid_map(ir_valid, b, h, w, device) if ir_srcs is not None
                     else None)
            depth_ok = (self._valid_map(depth_valid, b, h, w, device)
                        if depth_srcs is not None else None)
            weight = self.gate(lvl, rgb, a_ir, a_depth, text_up, ir_ok, depth_ok)
            weight = weight.to(rgb.dtype)  # [B,3,h,w]

            w_rgb = weight[:, 0:1]
            w_ir = weight[:, 1:2]
            w_depth = weight[:, 2:3]
            fused = w_rgb * rgb
            if a_ir is not None:
                fused = fused + w_ir * a_ir
            if a_depth is not None:
                fused = fused + w_depth * a_depth
            out.append(fused)

            if self.log_stats:
                with torch.no_grad():
                    # gate 统计一律在 fp32 下算, 避免 fp16 累加把 0.107 这种小权重抹平
                    wf = weight.float()
                    stats[f"gate_rgb_mean/l{lvl}"] = self._masked_mean(wf[:, 0:1], None)
                    stats[f"gate_ir_mean/l{lvl}"] = self._masked_mean(wf[:, 1:2], ir_ok)
                    stats[f"gate_depth_mean/l{lvl}"] = self._masked_mean(
                        wf[:, 2:3], depth_ok
                    )
                    # aux/RGB 强度比: 辅助候选**实际进入主路径**的强度(§9 第 4/5 行)。
                    # 只看权重会被「候选本身很小」骗过去, 所以按范数比。
                    base = (wf[:, 0:1] * rgb.float()).norm()
                    if a_ir is not None:
                        stats[f"aux_ir_ratio/l{lvl}"] = (
                            (wf[:, 1:2] * a_ir.float()).norm() / base.clamp_min(1e-6)
                        )
                    if a_depth is not None:
                        stats[f"aux_depth_ratio/l{lvl}"] = (
                            (wf[:, 2:3] * a_depth.float()).norm() / base.clamp_min(1e-6)
                        )

        self.last_stats = self._summarize(stats) if self.log_stats else None
        return out

    @staticmethod
    def _summarize(stats: dict) -> dict:
        """按 level 的明细 -> 再加一组跨 level 的平均值, 便于训练日志一行打印。"""
        out = dict(stats)
        for key in ("gate_rgb_mean", "gate_ir_mean", "gate_depth_mean",
                    "aux_ir_ratio", "aux_depth_ratio"):
            vals = [v for k, v in stats.items() if k.startswith(key + "/l")]
            if vals:
                out[key] = torch.stack(vals).mean()
        return out


# ======================================================================
#  Modality Dropout 与退化增强 (方案 §14)
# ======================================================================
class ModalityAugment(nn.Module):
    """训练期的模态 dropout / 退化增强, 让模型在 IR 或 Depth 质量下降时仍稳定。

    只在 self.training 下生效, 且逐**样本**随机(而非整 batch), 这样同一个 batch 内
    四种输入组合(RGB-only / RGB+IR / RGB+Depth / RGB+IR+Depth)都会出现。

    注意 "IR 或 Depth 缺失" 必须通过 valid flag 传递给 Fusion, 而不是只把输入置零 ——
    否则模型会把「整路置零」错误理解成真实的黑色 / 零距离信息。
    """

    def __init__(
        self,
        p_ir_drop: float = 0.0,
        p_depth_drop: float = 0.0,
        p_rgb_only: float = 0.0,
        p_ir_degrade: float = 0.0,
        p_depth_hole: float = 0.0,
        ir_noise_std: float = 0.05,
        ir_contrast_range=(0.7, 1.3),
        depth_hole_ratio: float = 0.25,
    ):
        super().__init__()
        self.p_ir_drop = p_ir_drop
        self.p_depth_drop = p_depth_drop
        self.p_rgb_only = p_rgb_only
        self.p_ir_degrade = p_ir_degrade
        self.p_depth_hole = p_depth_hole
        self.ir_noise_std = ir_noise_std
        self.ir_contrast_range = ir_contrast_range
        self.depth_hole_ratio = depth_hole_ratio

    @property
    def enabled(self) -> bool:
        return any(
            p > 0
            for p in (
                self.p_ir_drop, self.p_depth_drop, self.p_rgb_only,
                self.p_ir_degrade, self.p_depth_hole,
            )
        )

    def _bernoulli(self, p: float, b: int, device) -> torch.Tensor:
        if p <= 0:
            return torch.zeros(b, dtype=torch.bool, device=device)
        return torch.rand(b, device=device) < p

    def _degrade_ir(self, ir: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """噪声 + 对比度 + 模糊, 只作用于被选中的样本。"""
        if not deg.any():
            return ir
        b = ir.shape[0]
        factor = deg.to(ir.dtype).view(b, 1, 1, 1)

        lo, hi = self.ir_contrast_range
        contrast = 1.0 + factor * (torch.empty(b, 1, 1, 1, device=ir.device).uniform_(lo, hi) - 1.0)
        out = ir * contrast
        out = out + factor * torch.randn_like(ir) * self.ir_noise_std

        # 模糊只对少数样本做, 用循环避免给整个 batch 付 avg_pool 的代价
        for i in torch.nonzero(deg, as_tuple=False).flatten().tolist():
            k = 3
            blurred = F.avg_pool2d(out[i : i + 1], k, stride=1, padding=k // 2)
            out[i : i + 1] = blurred
        return out

    def _add_depth_holes(self, depth: torch.Tensor, hole: torch.Tensor) -> torch.Tensor:
        """随机挖一个矩形孔洞, 模拟真实深度图的缺失区域。

        孔洞直接置 0 —— DepthPreprocessor 会把 <= depth_min 的值判为 invalid,
        valid mask 与 D_norm / G_depth 会自动跟着变, 不需要额外传参。
        """
        if not hole.any():
            return depth
        b, _, h, w = depth.shape
        out = depth.clone()
        for i in torch.nonzero(hole, as_tuple=False).flatten().tolist():
            hh = max(1, int(h * self.depth_hole_ratio * float(torch.rand(1, device=depth.device).sqrt())))
            ww = max(1, int(w * self.depth_hole_ratio * float(torch.rand(1, device=depth.device).sqrt())))
            y0 = int(torch.randint(0, max(1, h - hh), (1,), device=depth.device))
            x0 = int(torch.randint(0, max(1, w - ww), (1,), device=depth.device))
            out[i, :, y0 : y0 + hh, x0 : x0 + ww] = 0
        return out

    @torch.no_grad()
    def forward(self, ir, depth, ir_valid=None, depth_valid=None):
        """返回 (ir, depth, ir_valid, depth_valid)。任何输入都允许为 None。"""
        if not (self.training and self.enabled):
            return ir, depth, ir_valid, depth_valid

        b = None
        for t in (ir, depth):
            if t is not None:
                b = t.shape[0]
                break
        if b is None:
            return ir, depth, ir_valid, depth_valid
        device = (ir if ir is not None else depth).device

        rgb_only = self._bernoulli(self.p_rgb_only, b, device)
        ir_drop = self._bernoulli(self.p_ir_drop, b, device) | rgb_only
        depth_drop = self._bernoulli(self.p_depth_drop, b, device) | rgb_only

        if ir is not None:
            keep = (~ir_drop).to(ir.dtype).view(b, 1, 1, 1)
            ir = ir * keep
            degrade = self._bernoulli(self.p_ir_degrade, b, device) & ~ir_drop
            ir = self._degrade_ir(ir, degrade)
            ir_valid = self._merge_valid(ir_valid, ~ir_drop, b, device)

        if depth is not None:
            keep = (~depth_drop).to(depth.dtype).view(b, 1, 1, 1)
            depth = depth * keep
            depth = self._add_depth_holes(depth, self._bernoulli(self.p_depth_hole, b, device) & ~depth_drop)
            depth_valid = self._merge_valid(depth_valid, ~depth_drop, b, device)

        return ir, depth, ir_valid, depth_valid

    @staticmethod
    def _merge_valid(valid, keep: torch.Tensor, b: int, device) -> torch.Tensor:
        if valid is None:
            valid = torch.ones(b, dtype=torch.bool, device=device)
        valid = valid.to(device).bool()
        # keep 是 [B], 而 valid 可以是 [B](逐样本)或 [B,1,H,W](逐像素,
        # DepthPreprocessor / 调用方都可能传这种)。后者直接 & [B] 会按「右对齐」
        # 广播, 把 [B] 当成最后一维 —— 只有 H*W 恰好等于 B 时才侥幸不报错。
        # V2 的 gate 是空间矩阵, 逐像素 valid 会被真正用上, 所以这条路径必须通。
        if valid.dim() > 1:
            keep = keep.view(b, *([1] * (valid.dim() - 1)))
        return valid & keep
