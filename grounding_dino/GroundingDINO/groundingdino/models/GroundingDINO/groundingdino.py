# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .multimodal_modules import (
    DepthEncoder,
    DepthPreprocessor,
    IREncoder,
    LanguageGuidedFusion,
    ModalityAugment,
    MultiScaleProjection,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss

# 方案 §12.3 的分组学习率默认值
DEFAULT_LR_GROUPS = {
    "ir": 1e-4,
    "depth": 1e-4,
    "fusion": 1e-4,
    "adapter": 1e-4,
    "head": 5e-5,
    "transformer": 1e-5,
    "rgb_stage34": 1e-5,
    "rgb_backbone": 1e-5,
    "text": 0.0,
}


def _as_aux_tensor(x):
    """辅助模态输入统一成 [B, C, H, W] 张量(NestedTensor 或裸 Tensor 都接受)。"""
    if isinstance(x, NestedTensor):
        return x.tensors
    return x


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        # ---------------- 多模态 (方案 §6 §7 §9 §10 §14) ----------------
        use_multimodal=False,
        ir_encoder_type="swin_t_warm_start",
        ir_backbone="swin_T_224_1k",
        ir_pretrain_img_size=224,
        ir_out_indices=(1, 2, 3),
        ir_use_checkpoint=False,
        depth_encoder_type="cnn_pyramid",
        depth_widths=(32, 64, 128, 256, 512),
        depth_depths=(0, 2, 2, 2, 2),
        depth_norm_mode="percentile",
        depth_min=0.0,
        depth_max=20000.0,
        depth_low_percentile=1.0,
        depth_high_percentile=99.0,
        depth_grad_scale=8.0,
        depth_hole_ratio=0.25,
        fusion_type="language_guided_residual",
        fusion_beta_init=0.0,
        fusion_gate_bias=-2.0,
        adapter_dim=64,
        modality_dropout_ir=0.0,
        modality_dropout_depth=0.0,
        modality_degrade_ir=0.0,
        modality_hole_depth=0.0,
        modality_rgb_only=0.0,
        unfreeze_encoder_layers_stage2=2,
        unfreeze_decoder_layers_stage2=2,
        unfreeze_rgb_swin_stage34_stage2=False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # ================= 多模态分支 (方案 §6 §7 §9 §14) =================
        self.use_multimodal = use_multimodal
        self.ir_encoder = None
        self.depth_encoder = None
        self.ir_proj = None
        self.depth_proj = None
        self.fusion = None
        # Fusion 只作用于 backbone 实际输出的 level 数(本配置是 3:H/8、H/16、H/32)。
        # 第 4 个 level(H/64)按方案 §5 由 input_proj 从原始 RGB 特征下采样得到, 不参与融合。
        self.num_fusion_levels = len(backbone.num_channels)
        assert self.num_fusion_levels <= num_feature_levels, (
            f"backbone 输出 {self.num_fusion_levels} 个 level, 超过 num_feature_levels={num_feature_levels}"
        )

        if use_multimodal:
            assert fusion_type == "language_guided_residual", f"未知 fusion_type {fusion_type!r}"

            if ir_encoder_type == "swin_t_warm_start":
                self.ir_encoder = IREncoder(
                    modelname=ir_backbone,
                    pretrain_img_size=ir_pretrain_img_size,
                    out_indices=tuple(ir_out_indices),
                    use_checkpoint=ir_use_checkpoint,
                )
            else:
                raise NotImplementedError(f"未知 ir_encoder_type {ir_encoder_type!r}")

            if depth_encoder_type == "cnn_pyramid":
                self.depth_encoder = DepthEncoder(
                    in_chans=3, widths=tuple(depth_widths), depths=tuple(depth_depths)
                )
            else:
                raise NotImplementedError(f"未知 depth_encoder_type {depth_encoder_type!r}")

            self.depth_preprocess = DepthPreprocessor(
                mode=depth_norm_mode,
                depth_min=depth_min,
                depth_max=depth_max,
                low_percentile=depth_low_percentile,
                high_percentile=depth_high_percentile,
                grad_scale=depth_grad_scale,
            )
            self.ir_proj = MultiScaleProjection(self.ir_encoder.out_channels, hidden_dim)
            self.depth_proj = MultiScaleProjection(self.depth_encoder.out_channels, hidden_dim)

            self.fusion = LanguageGuidedFusion(
                dim=hidden_dim,
                num_levels=self.num_fusion_levels,
                text_dim=hidden_dim,
                adapter_dim=adapter_dim,
                gate_bias=fusion_gate_bias,
                beta_init=fusion_beta_init,
            )

            self.modality_augment = ModalityAugment(
                p_ir_drop=modality_dropout_ir,
                p_depth_drop=modality_dropout_depth,
                p_rgb_only=modality_rgb_only,
                p_ir_degrade=modality_degrade_ir,
                p_depth_hole=modality_hole_depth,
                depth_hole_ratio=depth_hole_ratio,
            )

            # IR Swin 的 warm-start 必须挂在 load_state_dict 之后:
            # RGB Swin 的真实权重是加载 GroundingDINO checkpoint 时才到位的。
            self.register_load_state_dict_post_hook(GroundingDINO._warm_start_ir_hook)
        else:
            self.depth_preprocess = None
            self.modality_augment = None

        # Stage 2 的解冻范围 (方案 §11 §12.3)
        self._unfreeze_encoder_layers = int(unfreeze_encoder_layers_stage2)
        self._unfreeze_decoder_layers = int(unfreeze_decoder_layers_stage2)
        self._unfreeze_rgb_swin_stage34 = bool(unfreeze_rgb_swin_stage34_stage2)
        self.train_stage = None

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    # ==================================================================
    #  文本编码: 一次 forward 只跑一次 BERT (方案 §9)
    # ==================================================================
    def _encode_text(self, captions, device=None):
        """tokenize -> special-token mask -> BERT -> feat_map -> text_dict。

        抽成独立方法是为了让「一次 forward 只跑一次 BERT」成为结构性事实:
        Fusion 与 Transformer 共用这里返回的同一个 text_dict, 没有任何地方会重复编码。
        """
        if device is None:
            device = next(self.parameters()).device
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        return {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

    # ==================================================================
    #  多模态: 辅助分支编码 + 语言引导残差融合 (方案 §6 §7 §9)
    # ==================================================================
    def _aux_modality(self, kw, key, valid_key, samples):
        """取出一个辅助模态, 返回 (tensor, mask, valid) 或 (None, None, None)。

        支持裸 Tensor [B,1|3,H,W] 与 NestedTensor 两种输入; 两者都缺省时视为该模态缺失。
        """
        x = kw.get(key)
        if x is None:
            return None, None, None
        valid = kw.get(valid_key)
        if isinstance(x, NestedTensor):
            return x.tensors, x.mask, valid
        return x, samples.mask, valid

    @staticmethod
    def _all_invalid(valid, b, device) -> bool:
        """整 batch 该模态都无效 —— 此时没必要跑编码器, 直接走纯 RGB 路径。

        逐像素的 valid([B,1,H,W])按样本折叠成 [B] 再判断, 与
        LanguageGuidedFusion._as_valid 的语义保持一致; 否则逐像素 mask 会静默地
        绕过这个短路, 让「模态整路缺失」仍然付出一次编码器前向。
        """
        if valid is None:
            return False
        v = valid.to(device).bool()
        if v.dim() > 1:
            if v.shape[0] != b:
                return False
            v = v.reshape(b, -1).all(dim=-1)
        return bool(v.numel() == b and not v.any())

    def _encode_ir(self, ir, mask):
        """IR -> Conv2d(1,3,1) stem -> 独立 Swin-T -> 3 个 level, 每个 [B, 256, h, w]。"""
        feats = self.ir_encoder(_as_aux_tensor(ir), mask=mask)
        return self.ir_proj(feats)

    def _encode_depth(self, depth, valid):
        """Depth -> (D_norm, M_valid, G_depth) 3 通道 -> ConvNeXt 金字塔 -> [B, 256, h, w]。"""
        x_depth, _ = self.depth_preprocess(_as_aux_tensor(depth), valid=valid)
        return self.depth_proj(self.depth_encoder(x_depth))

    def _fuse_multimodal(self, srcs, samples, text_dict, kw):
        """在 input_proj 之后、进入 Transformer 之前, 把 IR / Depth 残差注入 RGB srcs。

        严格遵循方案 §17 的伪代码顺序:
            modality_augment -> 编码辅助模态 -> LanguageGuidedFusion
        两个模态都没传时**立即返回原 srcs**, 不做任何多余计算 —— 纯 RGB 调用路径
        (例如 run_grounding.py) 的行为与上游逐位一致。
        """
        ir, ir_mask, ir_valid = self._aux_modality(kw, "ir_samples", "ir_valid", samples)
        depth, _, depth_valid = self._aux_modality(kw, "depth_samples", "depth_valid", samples)
        if ir is None and depth is None:
            return srcs

        # 训练期的模态 dropout / 退火增强 (方案 §14)。eval 下是恒等变换。
        ir, depth, ir_valid, depth_valid = self.modality_augment(ir, depth, ir_valid, depth_valid)

        b = srcs[0].shape[0]
        device = srcs[0].device
        if ir is not None and self._all_invalid(ir_valid, b, device):
            ir = None
        if depth is not None and self._all_invalid(depth_valid, b, device):
            depth = None
        if ir is None and depth is None:
            # 整 batch 都被 modality dropout 掉了: 这一支走的就是「辅助模态缺失」的
            # 纯 RGB 路径, 与上游完全一致, 同时也省下一次辅助编码器的前向。
            return srcs

        ir_srcs = self._encode_ir(ir, ir_mask) if ir is not None else None
        depth_srcs = self._encode_depth(depth, depth_valid) if depth is not None else None

        return self.fusion(
            srcs,
            ir_srcs=ir_srcs,
            depth_srcs=depth_srcs,
            text_dict=text_dict,
            ir_valid=ir_valid,
            depth_valid=depth_valid,
        )

    # ==================================================================
    #  IR Swin 的 warm-start (方案 §6 方案 A)
    # ==================================================================
    def _warm_start_ir_encoder(self, incompatible_keys=None):
        """用 RGB Swin 的权重初始化 IR Swin。

        必须挂在 load_state_dict **之后**: RGB Swin 的真实权重是加载 GroundingDINO
        checkpoint 时才到位的, 构造期拷贝只能拷到随机初始化。

        实测 RGB Swin 与独立构建的 IR Swin 的 state_dict 键集完全相同(187 键)、形状
        全部一致, 所以这里就是一次普通的 load_state_dict, 不需要任何权重改形。
        """
        if self.ir_encoder is None:
            return None

        # checkpoint 自带 IR Swin 权重时(续训多模态模型)绝不能用 RGB 权重覆盖。
        if incompatible_keys is not None:
            missing = set(getattr(incompatible_keys, "missing_keys", None) or [])
            own = list(self.ir_encoder.body.state_dict().keys())
            if own and not all(f"ir_encoder.body.{k}" in missing for k in own):
                return None

        rgb_sd = self.backbone[0].state_dict()
        ir_sd = self.ir_encoder.body.state_dict()
        copied = {
            k: v for k, v in rgb_sd.items() if k in ir_sd and ir_sd[k].shape == v.shape
        }
        self.ir_encoder.body.load_state_dict(copied, strict=False)
        self._ir_warm_start_info = {
            "copied": len(copied),
            "skipped": sorted(k for k in ir_sd if k not in copied),
        }
        return self._ir_warm_start_info

    @staticmethod
    def _warm_start_ir_hook(module, incompatible_keys):
        module._warm_start_ir_encoder(incompatible_keys)

    # ==================================================================
    #  分阶段冻结 (方案 §11) 与分组学习率 (方案 §12.3)
    # ==================================================================
    @staticmethod
    def _set_group(modules, trainable: bool):
        """整组设置 requires_grad 与 train/eval。

        接受 Module 或裸 Parameter(如 `Transformer.level_embed`)—— 后者没有
        `.parameters()` 也没有 `.train()`, 必须单独处理。
        """
        for m in modules:
            if m is None:
                continue
            if isinstance(m, nn.Parameter):
                m.requires_grad_(trainable)
                continue
            for p in m.parameters():
                p.requires_grad_(trainable)
            m.train(trainable)

    def _module_groups(self) -> dict:
        """把模型拆成方案 §11 表格里的那些行, 便于整组冻结/解冻与分组学习率。"""
        enc = getattr(self.transformer, "encoder", None)
        dec = getattr(self.transformer, "decoder", None)
        swin = self.backbone[0]
        swin_stages = list(getattr(swin, "layers", []) or [])

        return {
            "text": [self.bert, self.feat_map],
            "rgb_swin": [swin],
            "rgb_swin_stage34": swin_stages[2:],
            "rgb_pos": [self.backbone[1]],
            "input_proj": [self.input_proj],
            "aux": [self.ir_encoder, self.depth_encoder, self.ir_proj,
                    self.depth_proj, self.fusion],
            # 检测头: 除了本模块自己的 bbox_embed / class_embed, 还要带上 two-stage 的
            # enc_out_* —— 它们挂在 transformer 上, 不显式列进来就永远不会被冻结。
            "head": [self.bbox_embed, self.class_embed,
                     getattr(self.transformer, "enc_out_bbox_embed", None),
                     getattr(self.transformer, "enc_out_class_embed", None),
                     getattr(self.transformer, "dec_out_bbox_embed", None),
                     getattr(self.transformer, "dec_out_class_embed", None)],
            "enc_base": list(enc.layers) if enc is not None else [],
            "enc_adapter": list(getattr(enc, "vision_adapters", []) or []) if enc is not None else [],
            "enc_text": list(getattr(enc, "text_layers", []) or []) if enc is not None else [],
            "enc_fusion": list(getattr(enc, "fusion_layers", []) or []) if enc is not None else [],
            "dec_base": list(dec.layers) if dec is not None else [],
            "dec_adapter": list(getattr(dec, "decoder_adapters", []) or []) if dec is not None else [],
            "dec_tail": ([getattr(dec, "norm", None), getattr(dec, "ref_point_head", None)]
                         if dec is not None else []),
            "transformer_misc": [getattr(self.transformer, "level_embed", None),
                                 getattr(self.transformer, "tgt_embed", None),
                                 getattr(self.transformer, "refpoint_embed", None),
                                 getattr(self.transformer, "enc_output", None),
                                 getattr(self.transformer, "enc_output_norm", None)],
        }

    def set_train_stage(self, stage: int):
        """按方案 §11 的表格设置 Stage 0/1/2/3 的冻结状态。

        两条硬约束:

        1. **绝不使用 torch.no_grad()**。方案 §11 明确警告: no_grad 会切断 Fusion 到 Loss
           的梯度。冻结只做 `requires_grad_(False)` + `eval()`。
        2. **IR/Depth 编码器、Fusion、Encoder/Decoder Adapter 在任何 stage 下都保持可训练**,
           它们是本次改造真正要学的东西。

        实现是「先全部冻结, 再按 stage 逐组解冻」, 因此结果与调用前的状态无关;
        每一组的 train/eval 也是显式设置的, 不依赖调用者先调过 model.train()。

        Args:
            stage: 0=RGB baseline, 1=multimodal warm-up, 2=partial unfreeze, 3=joint finetune

        Returns:
            dict: 便于训练脚本打印的统计信息
        """
        assert stage in (0, 1, 2, 3), f"train_stage 只能是 0/1/2/3, 得到 {stage}"
        self.train_stage = stage
        g = self._module_groups()

        # ---- 1) 全部冻结 ----
        for mods in g.values():
            self._set_group(mods, False)

        # ---- 2) 任何 stage 都训练的组 ----
        # 检测头 (方案 §11「BBox / text head: 训练/训练/训练」)
        self._set_group(g["head"], True)

        # Stage 0 = RGB baseline: 只训检测头, 辅助分支与 Adapter 也必须冻住。
        # 否则 beta 会从 0 漂走, F_new 不再等于 F_rgb, 「多模态模型在 stage 0 逐位
        # 等于 RGB 模型」这个对照前提就没了。
        if stage > 0:
            # 方案 §11「IR Encoder / Depth Encoder / Fusion / Encoder Adapter /
            # Decoder Adapter: 三个阶段都是训练」
            self._set_group(g["aux"], True)
            self._set_group(g["enc_adapter"], True)
            self._set_group(g["dec_adapter"], True)

        # 位置编码无参数, train/eval 无副作用, 只是保持一致
        self._set_group(g["rgb_pos"], True)

        n_enc = len(g["enc_base"])
        n_dec = len(g["dec_base"])

        if stage in (2, 3):
            keep_enc = n_enc if stage == 3 else min(self._unfreeze_encoder_layers, n_enc)
            keep_dec = n_dec if stage == 3 else min(self._unfreeze_decoder_layers, n_dec)
            # 「Encoder base layers: 冻结 → 最后 2 层可解冻 → 小学习率」
            self._set_group(g["enc_base"][n_enc - keep_enc:], True)
            self._set_group(g["dec_base"][n_dec - keep_dec:], True)
            # 解冻层配套的文本增强 / 特征融合层(逐层 clone, 按同样的后 N 层解冻)
            self._set_group(g["enc_text"][n_enc - keep_enc:], True)
            self._set_group(g["enc_fusion"][n_enc - keep_enc:], True)
            self._set_group(g["dec_tail"], True)
            self._set_group(g["transformer_misc"], stage == 3)

            if stage == 2 and self._unfreeze_rgb_swin_stage34:
                # 方案 §11「RGB Swin: 冻结 → Stage 3/4 可解冻」, 默认关闭
                self._set_group(g["rgb_swin_stage34"], True)

        if stage == 3:
            # 「RGB Swin: 小学习率联合微调」「RGB input_proj: 小学习率」
            self._set_group(g["rgb_swin"], True)
            self._set_group(g["input_proj"], True)

        # 文本编码器 (BERT + feat_map) 在所有 stage 都保持冻结 —— 方案 §11 里
        # BERT 三阶段都是「冻结 / 通常冻结」, config 的 lr_text 也默认 0。
        # 注意: 不冻结而给 lr=0 并不等价, AdamW 的 weight decay 仍会悄悄衰减 BERT。

        return self._describe_stage()

    def train(self, mode: bool = True):
        """覆写 nn.Module.train, 让冻结状态在每次 train() 之后依然成立。

        ⚠️ 为什么必须这么做: `nn.Module.train()` 会**递归**把所有子模块设成
        training=True。而 set_train_stage 的冻结只做 `requires_grad_(False)` + `eval()`,
        于是每个 epoch 例行的

            model.set_train_stage(1)
            model.train()

        会把「已冻结」的 RGB Swin 重新推回 training=True —— 参数确实不更新, 但 Swin 的
        DropPath(`drop_path_rate=0.2`, 本仓库 config 没有覆盖) 会重新生效, 冻结的主路径
        每次前向都变成随机的。这个坑很隐蔽, 所以在这里堵死: 只要设置过 train_stage,
        train(True) 之后就把 stage 逻辑重放一遍(它是幂等的: 先全部冻结再按 stage 解冻)。

        注意只覆写了 GroundingDINO 这一层; 直接对子模块调 train() 不受保护。
        """
        super().train(mode)
        if mode and getattr(self, "train_stage", None) is not None:
            self.set_train_stage(self.train_stage)
        return self

    def _describe_stage(self):
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        info = {
            "stage": self.train_stage,
            "num_trainable_tensors": len(trainable),
            "num_total_tensors": sum(1 for _ in self.parameters()),
        }
        self._stage_info = info
        return info

    def get_param_groups(self, lr_groups: Optional[dict] = None):
        """按方案 §12.3 返回分组参数列表, 可直接交给 optimizer。

        分组:
            ir / depth / fusion / adapter / head / transformer /
            rgb_stage34 / rgb_backbone / text

        每组内只收集 requires_grad=True 的参数, 冻结组不会出现在结果里。
        末尾做一次覆盖率检查: 任何被漏掉的参数都会抛错, 避免「悄悄不训练」。

        ⚠️ 两个前提, 否则分组会静默错位:

        1. **检测头在 named_parameters() 里不叫 `bbox_embed.*`。** 构造时执行了
           `self.transformer.decoder.bbox_embed = self.bbox_embed`, 而 `transformer`
           在 `_modules` 里排在 `bbox_embed` 前面, 于是去重后的名字是
           `transformer.decoder.bbox_embed.layers.0.weight`。因此这里必须按**子串**
           匹配而不是 `startswith`, 否则 head 组会是空的、检测头被按 1e-5 训练。
        2. **Swin 与 position embedding 同在 `backbone` 下**, 所以 `rgb_stage34`
           只能按 `backbone.0.layers.<i>.` (i=2,3) 的前缀切出来。
        """
        lrs = dict(DEFAULT_LR_GROUPS)
        if lr_groups:
            lrs.update({k: v for k, v in lr_groups.items() if v is not None})

        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]

        groups = {}
        groups["ir"] = [t for t in named
                        if t[0].startswith(("ir_encoder.", "ir_proj."))]
        groups["depth"] = [t for t in named
                           if t[0].startswith(("depth_encoder.", "depth_proj."))]
        groups["fusion"] = [t for t in named if t[0].startswith("fusion.")]

        aux = {n for g in ("ir", "depth", "fusion") for n, _ in groups[g]}
        groups["adapter"] = [t for t in named
                             if t[0] not in aux
                             and ("vision_adapters" in t[0] or "decoder_adapters" in t[0])]
        adapter = {n for n, _ in groups["adapter"]}

        groups["head"] = [t for t in named
                          if t[0] not in aux and t[0] not in adapter
                          and ("bbox_embed" in t[0] or "class_embed" in t[0])]
        head = {n for n, _ in groups["head"]}

        groups["text"] = [t for t in named
                          if t[0] not in aux | adapter | head
                          and t[0].startswith(("bert.", "feat_map.", "tokenizer."))]
        text = {n for n, _ in groups["text"]}

        def is_rgb_stage34(n):
            # backbone.0 是 Swin, 其 layers 4 个 stage; stage3/4 指 layers[2:] (H/16、H/32)
            if not n.startswith("backbone.0.layers."):
                return False
            return n.split(".")[3] in ("2", "3")

        used = aux | adapter | head | text
        groups["rgb_stage34"] = [t for t in named if t[0] not in used and is_rgb_stage34(t[0])]
        stage34 = {n for n, _ in groups["rgb_stage34"]}

        groups["rgb_backbone"] = [t for t in named
                                  if t[0] not in used | stage34
                                  and t[0].startswith(("backbone.", "input_proj."))]
        backbone = {n for n, _ in groups["rgb_backbone"]}

        # 剩下的 (transformer 主体) 归入 transformer 组
        groups["transformer"] = [t for t in named
                                 if t[0] not in used | stage34 | backbone]

        covered = used | stage34 | backbone | {n for n, _ in groups["transformer"]}
        missed = [n for n, _ in named if n not in covered]
        assert not missed, f"以下可训练参数没有被分到任何学习率组: {missed[:10]}"

        out = []
        for name in ("ir", "depth", "fusion", "adapter", "head", "transformer",
                     "rgb_stage34", "rgb_backbone", "text"):
            params = [p for _, p in groups[name]]
            if not params:
                continue
            out.append({"name": name, "lr": lrs[name], "params": params})
        return out

    def set_image_tensor(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        self.features, self.poss = self.backbone(samples)

    def unset_image_tensor(self):
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self,'poss'):
            del self.poss 

    def set_image_features(self, features , poss):
        self.features = features
        self.poss = poss

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        # encoder texts —— 整个 forward 里 BERT 只在这里跑一次, 之后 text_dict 由
        # Fusion 与 Transformer 共用 (方案 §9)
        text_dict = self._encode_text(captions, device=samples.device)

        # import ipdb; ipdb.set_trace()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if not hasattr(self, 'features') or not hasattr(self, 'poss'):
            self.set_image_tensor(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(self.features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        # ---- 多模态融合 (方案 §17): 在 input_proj 之后、进入 Transformer 之前 ----
        # 只改写前 num_fusion_levels 个 level 的**数值**; masks / poss / 空间尺度 / padding
        # 一律保持 RGB 原样, 保证 Transformer 与检测头的接口不变。
        if self.use_multimodal:
            srcs = self._fuse_multimodal(srcs, samples, text_dict, kw)

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](self.features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                self.poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, self.poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # ---- 训练专用输出 (方案: 数据管道与训练脚本阶段新增) ----
        # 仅在传入 targets 时附加, 推理路径(targets is None)的 out 键集与上游逐位一致,
        # 所以 run_grounding.py / demo 的调用不受任何影响。
        if targets is not None:
            # # for intermediate outputs
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
            # criterion 需要知道哪些 token 槽位是真实文本(用于置正样本标签)。
            # 从 text_dict 直接取, 避免 criterion 再 tokenize 一遍造成口径漂移。
            #
            # ⚠️ 宽度要补齐: text_dict 里的 mask 长度是「本 batch 最长 caption 的 token 数」
            # (tokenizer 用 padding="longest"), 而 pred_logits 的最后一维是 max_text_len ——
            # ContrastiveEmbed 把真实 token 放在**前 n 个槽位**、其余填 -inf。两者宽度不同,
            # 不补齐 criterion 会直接索引越界。
            out["text_token_mask"] = self._pad_token_mask(
                text_dict["text_token_mask"], outputs_class.shape[-1]
            )

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
        unset_image_tensor = kw.get('unset_image_tensor', True)
        if unset_image_tensor:
            self.unset_image_tensor() ## If necessary
        return out

    @staticmethod
    def _pad_token_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
        """把 [B, n_tok] 的 token mask 补成 [B, width](width = pred_logits 的 token 维)。

        ContrastiveEmbed 的槽位布局是「真实 token 在前, 其余填 -inf」, 所以补出来的
        尾部一律 False(padding), 与 -inf 的位置完全对应。
        """
        n = min(mask.shape[1], width)
        if mask.shape[1] == width:
            return mask
        full = mask.new_zeros((mask.shape[0], width))
        full[:, :n] = mask[:, :n]
        return full

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        # ---------------- 多模态 (全部用 getattr 兜底, 旧 config 仍可正常构建) ----------------
        use_multimodal=getattr(args, "use_multimodal", False),
        ir_encoder_type=getattr(args, "ir_encoder_type", "swin_t_warm_start"),
        ir_backbone=getattr(args, "ir_backbone", "swin_T_224_1k"),
        ir_pretrain_img_size=getattr(args, "ir_pretrain_img_size", 224),
        ir_out_indices=tuple(getattr(args, "ir_out_indices", (1, 2, 3))),
        ir_use_checkpoint=getattr(args, "ir_use_checkpoint", False),
        depth_encoder_type=getattr(args, "depth_encoder_type", "cnn_pyramid"),
        depth_widths=tuple(getattr(args, "depth_widths", (32, 64, 128, 256, 512))),
        depth_depths=tuple(getattr(args, "depth_depths", (0, 2, 2, 2, 2))),
        depth_norm_mode=getattr(args, "depth_norm_mode", "percentile"),
        depth_min=getattr(args, "depth_min", 0.0),
        depth_max=getattr(args, "depth_max", 20000.0),
        depth_low_percentile=getattr(args, "depth_low_percentile", 1.0),
        depth_high_percentile=getattr(args, "depth_high_percentile", 99.0),
        depth_grad_scale=getattr(args, "depth_grad_scale", 8.0),
        depth_hole_ratio=getattr(args, "depth_hole_ratio", 0.25),
        fusion_type=getattr(args, "fusion_type", "language_guided_residual"),
        fusion_beta_init=getattr(args, "fusion_beta_init", 0.0),
        fusion_gate_bias=getattr(args, "fusion_gate_bias", -2.0),
        adapter_dim=getattr(args, "adapter_dim", 64),
        modality_dropout_ir=getattr(args, "modality_dropout_ir", 0.0),
        modality_dropout_depth=getattr(args, "modality_dropout_depth", 0.0),
        modality_degrade_ir=getattr(args, "modality_degrade_ir", 0.0),
        modality_hole_depth=getattr(args, "modality_hole_depth", 0.0),
        modality_rgb_only=getattr(args, "modality_rgb_only", 0.0),
        unfreeze_encoder_layers_stage2=getattr(args, "unfreeze_encoder_layers_stage2", 2),
        unfreeze_decoder_layers_stage2=getattr(args, "unfreeze_decoder_layers_stage2", 2),
        unfreeze_rgb_swin_stage34_stage2=getattr(args, "unfreeze_rgb_swin_stage34_stage2", False),
    )

    return model

