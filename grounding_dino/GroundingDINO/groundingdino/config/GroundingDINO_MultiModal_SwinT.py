# ------------------------------------------------------------------------
# GroundingDINO RGB / IR / Depth 多模态配置 (Swin-T)
# ------------------------------------------------------------------------
# 基线是 GroundingDINO_SwinT_OGC.py, 此处只**追加**多模态相关参数,
# 原有键一律保持原值 —— 这样 Stage 0 的 RGB baseline 与多模态模型可以在
# 完全相同的检测超参下对比(消融矩阵 E0 vs E4/E5/E6)。
#
# 用法:
#   from groundingdino.util.slconfig import SLConfig
#   args = SLConfig.fromfile("groundingdino/config/GroundingDINO_MultiModal_SwinT.py")
#   model = build_model(args)
#   model(images, captions=[...],
#         ir_samples=ir, depth_samples=depth, ir_valid=iv, depth_valid=dv)
# ------------------------------------------------------------------------

# ====================== 以下与 GroundingDINO_SwinT_OGC.py 完全一致 ======================
batch_size = 1
modelname = "groundingdino"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True

# ====================== 多模态总开关 ======================
# False 时模型与上游 GroundingDINO 逐位等价, 不构建任何辅助分支。
use_multimodal = True

# ====================== IR 分支 (方案 §6 方案 A) ======================
# "swin_t_warm_start": Conv2d(1,3,1) stem(权重 1/3) + 独立 Swin-T,
#                      权重由 RGB Swin 在 load_state_dict 后 warm-start。
ir_encoder_type = "swin_t_warm_start"
ir_backbone = "swin_T_224_1k"
ir_pretrain_img_size = 224
# 训练时这个**必须**开。stage 1/2/3 里 IR Swin 是全量可训练的, 在 1333x750 分辨率上
# 不挂 gradient checkpointing 的激活值开销与「再训一个 RGB Swin」同级 —— 是显存杀手。
# 推理(eval)不受影响, 只多一点点时间。
ir_use_checkpoint = True

# ====================== Depth 分支 (方案 §7) ======================
depth_encoder_type = "cnn_pyramid"
# "percentile": 逐样本用有效像素的 p1/p99 作归一化区间(默认)。
#   实测本数据集的 invalid 占比 2.3%~61.4%、每图值域差异极大, 固定区间会损失一半动态范围。
# "fixed": 用下面的 depth_min / depth_max。
depth_norm_mode = "percentile"
depth_min = 0.0
depth_max = 20000.0
depth_low_percentile = 1.0
depth_high_percentile = 99.0
# |∇x| + |∇y| 在归一化深度上的理论最大值(±4 的 Sobel 正权重之和), 用作边缘图的固定归一化尺度
depth_grad_scale = 8.0
depth_hole_ratio = 0.25

# ====================== Fusion (方案 §9 / V2 §5) ======================
# 两版 Fusion 二选一, 接口完全一致(见 groundingdino.py 的 _fuse_multimodal):
#   "local_cross_attention_spatial_gate"  第二版主方案: 局部 cross-attention 候选
#                                         + B x 3 x H x W 空间矩阵 gate
#   "language_guided_residual"            第一版: beta 标量控制的残差注入
#                                         (V2 §11 消融矩阵的 V2-E4 用它做对照)
fusion_type = "local_cross_attention_spatial_gate"

# ---- 第二版专属 (V2 §6 §14) ----
# 局部 cross-attention 的窗口边长。配准良好时 3 就够; 有轻微错位用 5(默认)。
# 边界定位差就往上调(§12「边界定位差」一行)。
fusion_window_size = 5
# 与 hidden_dim=256 对齐, 每头 32 维
fusion_num_heads = 8
# 只有 spatial_softmax 一种实现: 输出 B x 3 x H x W, 通道依次是
# RGB / IR-attended / Depth-attended
fusion_gate_type = "spatial_softmax"
# gate bias 初值 [rgb, ir, depth]。softmax([2,0,0]) = [0.786, 0.107, 0.107]。
# ⚠️ 第二版**不再**零初始化: 辅助模态从第一步就拿到非零权重入口(V2 §5.3)。
# 性能比 RGB 明显下降时先上调到 3.0(softmax -> [0.91, 0.045, 0.045]), 更保守。
fusion_gate_rgb_bias = 2.0
fusion_gate_aux_bias = 0.0
# gate 的输入是否包含句子级文本向量 T_global 的广播
fusion_gate_use_text = True
# 记录 gate_rgb_mean / gate_ir_mean / gate_depth_mean / aux_ir_ratio / aux_depth_ratio
# (V2 §9)。这是判断「辅助模态是否真的被用上」的唯一依据, 不要关。
fusion_log_stats = True

# ---- 第一版专属(改回 fusion_type="language_guided_residual" 时才生效) ----
# 输出侧零初始化: beta=0 + Adapter 末层为 0, 训练起点上 F_new == F_rgb。
# 注意这个开关对第二版**不适用**, 第二版靠 gate bias 而非零初始化。
fusion_zero_init = False
# 每个 level 一个可学习标量 beta 的初值。0 时模型严格等于 RGB 模型;
# 若发现辅助支路起步太慢可改成 1e-3。
fusion_beta_init = 0.0
# 第一版空间 gate(sigmoid)的 bias 初值, sigmoid(-2) ≈ 0.12
fusion_gate_bias = -2.0

# ====================== Adapter (方案 §10) ======================
# 插入位置: Encoder 在 BiAttentionBlock 与 Deformable encoder layer 之间,
#           Decoder 在每个 decoder layer 之后、iter-update 之前。
encoder_adapter = True
decoder_adapter = True
adapter_dim = 64
adapter_dropout = 0.0

# ====================== 冻结与训练阶段 (方案 §11 §12.3) ======================
# train_stage: 0=RGB baseline, 1=multimodal warm-up, 2=partial unfreeze, 3=joint finetune
# 建成模型后调用 model.set_train_stage(args.train_stage) 生效。
train_stage = 1
train_detection_head = True
freeze_text_encoder_stage1 = True
freeze_rgb_backbone_stage1 = True
# Stage 2 解冻的后部层数
unfreeze_encoder_layers_stage2 = 2
unfreeze_decoder_layers_stage2 = 2
unfreeze_rgb_swin_stage34_stage2 = False

# 分组学习率 (方案 §12.3)
lr_ir = 1e-4
lr_depth = 1e-4
lr_fusion = 1e-4
lr_adapter = 1e-4
lr_head = 5e-5
lr_transformer = 1e-5
lr_rgb_stage34 = 1e-5
lr_text = 0.0

# ====================== 模态 dropout 与退化 (方案 §14) ======================
modality_dropout_ir = 0.15
modality_dropout_depth = 0.15
modality_degrade_ir = 0.10
modality_hole_depth = 0.10
modality_rgb_only = 0.10

# 方案 §13: 蒸馏与 gate 正则要等 baseline 稳定后再开, 第一版关闭
use_teacher_distillation = False
