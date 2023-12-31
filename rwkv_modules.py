import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import math
import os, types

rwkv_config_1 = types.SimpleNamespace()
rwkv_config_1.datafile = os.path.join("data", "enwik8")

rwkv_config_1.batch_size = 1
rwkv_config_1.ctx_len = 1024
rwkv_config_1.lr = 0.00001
rwkv_config_1.vocab_size = vocab_size = 1000

rwkv_config_1.betas = (0.9, 0.999)
rwkv_config_1.eps = 1e-8
rwkv_config_1.block_size = 150
rwkv_config_1.device = "cuda" if torch.cuda.is_available() else "cpu"
rwkv_config_1.embd_pdrop = 0.1
rwkv_config_1.n_embd = 512
rwkv_config_1.n_layer = 12
rwkv_config_1.max_timestep = 5409
rwkv_config_1.model_name = "rwkv_demo"
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, x):
        ctx.save_for_backward(x)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        # factor 是一个常数，用于控制 L2 正则化的强度。它通过除以 y 的总元素个数 B * T 来对梯度进行缩放。
        factor = 1e-4 / (x.shape[0] * x.shape[1])
        """ 
         返回一个形状为 [B, T, 1] 的张量 maxx，其中每个元素是该样本对应的输出 x 中的最大值。
         同时，ids 是一个形状为 [B, T, 1] 的张量，其中每个元素是最大值 maxx 对应的索引。
           """
        maxx, ids = torch.max(x, -1, keepdim=True)

        """ 
        gx 是一个与 x 相同形状的全零张量。
        然后，使用 scatter_ 函数，将 maxx * factor 分散到 gx 的每个样本的最大值所在的位置上。
        也就是说，对于每个样本，将其最大值处的梯度设置为 maxx * factor，其余位置的梯度保持为零。
          """

        gx = torch.zeros_like(x)
        gx.scatter_(-1, ids, maxx * factor)
        return (grad_output, gx)


# 定义模型和优化器


local_env = os.environ.copy()
local_env["PATH"] = r"D:\Github\rwkv_cu118\Scripts;" + local_env["PATH"]
os.environ.update(local_env)

## 将wkv融入torch计算流中
from torch.utils.cpp_extension import load

T_MAX = 1024
wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=
                ['-res-usage', '--use_fast_math', '-O3', '--maxrregcount=60', '-Xptxas=-O3', f'-DTmax={T_MAX}'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w = -torch.exp(w.float().contiguous())
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        return y

        # if '32' in os.environ['RWKV_FLOAT_MODE']:
        #     return y
        # elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
        #     return y.half()
        # elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
        #     return y.bfloat16()

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return (None, None, None, gw, gu, gk, gv)
        #
        # if '32' in os.environ['RWKV_FLOAT_MODE']:
        #     return (None, None, None, gw, gu, gk, gv)
        # elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
        #     return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        # elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
        #     return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

class RWKV_TimeMix(torch.jit.ScriptModule):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id  # 当前layer id
        self.ctx_len = rwkv_config_1.ctx_len  # 最长文本长度
        self.n_embd = rwkv_config_1.n_embd  # hidden_state 维度

        # todo 附录D中TimeMix的位置编码w、u(mu)、u 的初始化计算方法
        with torch.no_grad():  # fancy init
            """
            layer_id 是 w_i的 l
            config.n_layer 是 w_i的 L
            """
            ratio_0_to_1 = (layer_id / (rwkv_config_1.n_layer - 1))  # 0 to 1   w的  l / (L - 1)

            ratio_1_to_almost0 = (1.0 - (layer_id / rwkv_config_1.n_layer))  # 1 to ~0   u(mu)的  1-（l/L）

            # fancy time_decay
            decay_speed = torch.ones(self.n_embd)  # 维度的位置编码 [hidden_state_size]
            for h in range(self.n_embd):  # 按隐藏维度循环每一个位置
                """
                h 对应 （14） 公式中w_i的i 
                attn_sz - 1  对应 （14） 公式中w_i的 (d-1)
                ratio_0_to_1  对应 （14） 公式中w_i的 l / (L - 1)
                """
                decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first 对应 论文中的bonus
            """
            [(i + 1) % 3 - 1 for i in range(attn_sz)] 对应 （14） 公式中u 的 ((i+1) mod 3) -1

            zigzag对应 （14） 公式中u 的 0.5 * ((i+1) mod 3) -1

            self.time_first 对应 （14） 公式中u
            """
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

            # fancy time_mix 对应公式中的(11-13)
            x = torch.ones(1, 1, rwkv_config_1.n_embd)
            for i in range(rwkv_config_1.n_embd):
                """
                config.n_embd 对应 s
                """
                x[0, 0, i] = i / rwkv_config_1.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))  # 对应 U(mu)_ki
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)  # 对应 U(mu)_Vi
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))  # 对应 U(mu)_ri
        # todo 平移操作利于生成 X_t-1
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # 定义 Wr Wk Wv
        self.key = nn.Linear(rwkv_config_1.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(rwkv_config_1.n_embd, self.n_embd, bias=False)
        self.receptance = nn.Linear(rwkv_config_1.n_embd, self.n_embd, bias=False)
        # 定义 Wo
        self.output = nn.Linear(self.n_embd, rwkv_config_1.n_embd, bias=False)
        # todo 不懂
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    @torch.jit.script_method  # 声明是一个静态图，预先搭建好计算图，往里面加数据即可
    def jit_func(self, x):
        """C++ 调用"""
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)  # X_t-1
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 公式 (12) 中的 括号部分
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)  # 公式 (13) 中的 括号部分
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 公式 (11) 中的 括号部分
        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)  # 公式 (12) 中的K_t
        v = self.value(xv)  # 公式 (13) 中的V_t
        r = self.receptance(xr)  # 公式 (11) 中的R_t
        sr = torch.sigmoid(r)  # 公式 (15) 中的sigmoid_Rt
        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)  <=>   batch_size sentence_len hidden_size
        sr, k, v = self.jit_func(x)
        """
        RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v) 对应 公式 (14) 中的wkv_t
        rwkv 对应 公式 (15) 中的 小括号内容
        """
        rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)  # 对应公式 (15)
        return rwkv


class RWKV_ChannelMix(torch.jit.ScriptModule):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id  # layer id

        # 平移
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            # todo 参考 time mix中的 的位置编码u(mu) 的初始化计算方法

            ratio_1_to_almost0 = (1.0 - (layer_id / rwkv_config_1.n_layer))  # 1 to ~0

            x = torch.ones(1, 1, rwkv_config_1.n_embd)
            for i in range(rwkv_config_1.n_embd):
                x[0, 0, i] = i / rwkv_config_1.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        hidden_sz = 4 * rwkv_config_1.n_embd
        self.key = nn.Linear(rwkv_config_1.n_embd, hidden_sz, bias=False)  # 对应公式(17) 中的 W_k
        self.receptance = nn.Linear(rwkv_config_1.n_embd, rwkv_config_1.n_embd, bias=False)  # 对应公式(16) 中的 W_r
        self.value = nn.Linear(hidden_sz, rwkv_config_1.n_embd, bias=False)  # 对应公式(18) 中的 W_v

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    @torch.jit.script_method
    def forward(self, x):
        xx = self.time_shift(x)  # 公式（16-17）中的 X_t-1
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 公式（17）中的 括号部分
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 公式（16）中的 括号部分

        k = self.key(xk)  # 公式（17）中的结果
        k = torch.square(torch.relu(k))  # 公式（18）中的 max(k_t,0)的平方
        kv = self.value(k)  # 公式（18）中的第二个括号部分

        rkv = torch.sigmoid(self.receptance(xr)) * kv  # 公式（18）中
        return rkv

class Block(nn.Module):
    """一个RWKV块"""

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id  # 当前layer的id
        self.ln1 = nn.LayerNorm(rwkv_config_1.n_embd)
        self.ln2 = nn.LayerNorm(rwkv_config_1.n_embd)
        self.Time_mix = RWKV_TimeMix(layer_id)
        self.Channel_mix = RWKV_ChannelMix(layer_id)

    def forward(self, x):

        x = x + self.Time_mix(self.ln1(x))
        x = x + self.Channel_mix(self.ln2(x))
        return x

class decisionRWKV(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.step = 0
        self.vocab_size = vocab_size

        self.ctx_len = rwkv_config_1.ctx_len

        self.emb = nn.Embedding(self.vocab_size, rwkv_config_1.n_embd)
        self.ln_in = nn.LayerNorm(rwkv_config_1.n_embd)

        # RWKV 模块层
        self.blocks = nn.Sequential(*[Block(i) for i in range(rwkv_config_1.n_layer)])

        self.ln_out = nn.LayerNorm(rwkv_config_1.n_embd)
        self.linear = nn.Linear(rwkv_config_1.n_embd, self.vocab_size, bias=False)
        # build modules
        self.global_timestep_encoding = nn.Embedding(rwkv_config_1.max_timestep, rwkv_config_1.n_embd)
        self.context_position_encoding = nn.Embedding(rwkv_config_1.block_size, rwkv_config_1.n_embd)
        self.dropout = nn.Dropout(rwkv_config_1.embd_pdrop)
        self.block_loop = nn.ModuleList([Block(rwkv_config_1.vocab_size) for _ in range(rwkv_config_1.n_layer)])
        self.norm = nn.LayerNorm(rwkv_config_1.n_embd)
        self.lm_head = nn.Linear(rwkv_config_1.n_embd, rwkv_config_1.vocab_size, bias=False)

        # initialize weights
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    \

    def forward(self, rtgs_emb, states_emb, actions_emb, timesteps):
        batch_size = states_emb.shape[0]
        actual_step_size = states_emb.shape[1]

        token_emb = torch.zeros(
            (batch_size, actual_step_size*3, rwkv_config_1.n_embd),
            dtype=torch.float32,
            device=states_emb.device)
        token_emb[:,::3,:] = rtgs_emb
        token_emb[:,1::3,:] = states_emb
        if actions_emb is not None:
            token_emb[:,2::3,:] = actions_emb

        timestep_start = torch.repeat_interleave(timesteps[:,0].unsqueeze(dim=-1), actual_step_size*3, dim=-1) # (batch_size, actual_step_size*3)
        pos_global = self.global_timestep_encoding(timestep_start)
        context_position = torch.arange(actual_step_size*3, device=states_emb.device).repeat(batch_size,1) # (batch_size, actual_step_size*3)
        pos_relative = self.context_position_encoding(context_position)
        pos_emb = pos_global + pos_relative

        x = self.dropout(token_emb + pos_emb)

        for block in self.block_loop:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)
        # only get predictions from states
        logits = logits[:,1::3,:]

        return logits
