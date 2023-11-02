import torch
import torch.nn as nn
from torch.nn import functional as F
import atari_py
import cv2
import matplotlib.pyplot as plt
import math
import torch
import io
import math
import numpy as np
import os
import gzip
import pickle
from PIL import Image
import PIL
import types
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ale = atari_py.ALEInterface()
ale.loadROM(atari_py.get_game_path("pong"))
ale.reset_game()
available_actions = ale.getMinimalActionSet()
action_dict = dict([i, e] for i, e in enumerate(available_actions))
train_loss = []

rwkv_config_1 = types.SimpleNamespace()
rwkv_config_1.datafile = os.path.join("data", "enwik8")

rwkv_config_1.batch_size = 1
rwkv_config_1.ctx_len = 1024
rwkv_config_1.vocab_size = vocab_size = len(action_dict)
rwkv_config_1.lr = 0.00001
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

        factor = 1e-4 / (x.shape[0] * x.shape[1])

        maxx, ids = torch.max(x, -1, keepdim=True)


        gx = torch.zeros_like(x)
        gx.scatter_(-1, ids, maxx * factor)
        return (grad_output, gx)


local_env = os.environ.copy()
local_env["PATH"] = r"D:\Github\rwkv_cu118\Scripts;" + local_env["PATH"]
os.environ.update(local_env)

from torch.utils.cpp_extension import load


T_MAX = 1024
wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=
                ['-res-usage', '--use_fast_math', '-O3','--maxrregcount=60', '-Xptxas=-O3', f'-DTmax={T_MAX}'])

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        print(T)
        print("test")
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

        with torch.no_grad():  # fancy init

            ratio_0_to_1 = (layer_id / (rwkv_config_1.n_layer - 1))  # 0 to 1   w的  l / (L - 1)

            ratio_1_to_almost0 = (1.0 - (layer_id / rwkv_config_1.n_layer))  # 1 to ~0   u(mu)的  1-（l/L）

            # fancy time_decay
            decay_speed = torch.ones(self.n_embd)  # 维度的位置编码 [hidden_state_size]
            for h in range(self.n_embd):  # 按隐藏维度循环每一个位置

                decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first 对应 论文中的bonus

            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

            # fancy time_mix 对应公式中的(11-13)
            x = torch.ones(1, 1, rwkv_config_1.n_embd)
            for i in range(rwkv_config_1.n_embd):

                x[0, 0, i] = i / rwkv_config_1.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))  # 对应 U(mu)_ki
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)  # 对应 U(mu)_Vi
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))  # 对应 U(mu)_ri


        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # 定义 Wr Wk Wv
        self.key = nn.Linear(rwkv_config_1.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(rwkv_config_1.n_embd, self.n_embd, bias=False)
        self.receptance = nn.Linear(rwkv_config_1.n_embd, self.n_embd, bias=False)

        # 定义 Wo
        self.output = nn.Linear(self.n_embd, rwkv_config_1.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    @torch.jit.script_method
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

        rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)

        rwkv = self.output(rwkv)  # 对应公式 (15)
        print("##")
        print(rwkv)
        print(rwkv.size())

        return rwkv


class RWKV_ChannelMix(torch.jit.ScriptModule):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id  # layer id

        # 平移
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix


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
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

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

    def configure_optimizers(self):
        no_decay = set()

        for mn, m in self.named_modules():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.Adam(optim_groups, lr=rwkv_config_1.lr, betas=rwkv_config_1.betas,
                                     eps=rwkv_config_1.eps)

        return optimizer
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


class Embeddings_for_Atari(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.state_embedding = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, config.n_embd),
            nn.Tanh()
        )
        self.action_embedding = nn.Sequential(
            nn.Embedding(config.vocab_size, config.n_embd),
            nn.Tanh()
        )
        self.rtg_embedding = nn.Sequential(
            nn.Linear(1, config.n_embd),
            nn.Tanh()
        )

        # initialize weights
        self.apply(self._init_weights)

    # see karpathy/minGPT for weight's initilization in OpenAI GPT
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, rtgs, states, actions):
        ### inputs
        # rtgs        : (batch_size, step_size, 1)
        # states      : (batch_size, step_size, 4, 84, 84)
        # actions     : (batch_size, step_size)

        ### outputs
        # rtgs_emb    : (batch_size, step_size, n_embd)
        # states_emb  : (batch_size, step_size, n_embd)
        # actions_emb : (batch_size, step_size, n_embd)

        rtgs_emb = self.rtg_embedding(rtgs)

        states_shp = states.reshape(-1, 4, 84, 84)
        states_emb = self.state_embedding(states_shp)
        states_emb = states_emb.reshape(states.shape[0], states.shape[1], states_emb.shape[1])

        if actions is None:
            actions_emb = None
        else:
            actions_emb = self.action_embedding(actions)

        return rtgs_emb, states_emb, actions_emb

source_data_dir = "downloaded_game_data/Pong/1/replay_logs"
dest_dir = "game_dataset"
step_size = 50
chunk_size = 10000  # meta data will be chunked by each 10000 rows
stack_size = 4      # frame stacking size (So stack_size - 1 frames are added in each states)
max_timesteps = 5409

gpt = decisionRWKV(rwkv_config_1.vocab_size).to(device)
emb = Embeddings_for_Atari(rwkv_config_1).to(device)

####  Play game with not-trained agent

import atari_py
from collections import deque
import random

class EnvWrapper():
    def __init__(self, device, stack_size):
        self.device = device
        self.stack_size = stack_size

        self.ale = atari_py.ALEInterface()
        self.ale.setInt("random_seed", 123)
        self.ale.setInt("max_num_frames_per_episode", 108e3)
        self.ale.setFloat("repeat_action_probability", 0)
        self.ale.setInt("frame_skip", 0)
        self.ale.setBool("color_averaging", False)
        self.ale.loadROM(atari_py.get_game_path("pong"))

        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))

        self.state_buffer = deque([], maxlen=stack_size)

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def reset(self):
        for _ in range(self.stack_size):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))
        self.ale.reset_game()
        # Perform up to 30 random no-ops before starting
        for _ in range(random.randrange(30)):
            self.ale.act(0)  # Assumes raw action 0 is always no-op
            if self.ale.game_over():
                self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done
from torch.nn import functional as F

@torch.no_grad()
def run_game(target_rtg, emb, gpt, device, stack_size, max_timesteps):
    #model.train(False)

    env = EnvWrapper(device, stack_size)

    done = True

    # reset
    state = env.reset()
    state = state.type(torch.float32).to(device).unsqueeze(0)
    rtgs = [target_rtg]

    # pick up next action
    rtgs_emb, states_emb, actions_emb = emb(
        rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1).type(torch.float32),
        states=state.unsqueeze(0),
        actions=None,
    )
    logits = gpt(
        rtgs_emb=rtgs_emb,
        states_emb=states_emb,
        actions_emb=actions_emb,
        timesteps=torch.zeros((1, 1), dtype=torch.int64).to(device),
    )
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    sampled_action = torch.multinomial(probs, num_samples=1)

    time_start = 0
    all_states = state
    actions = []
    reward_sum = 0
    while True:
        action = sampled_action.cpu().numpy()[0,-1]
        actions += [sampled_action]
        state, reward, done = env.step(action)
        reward_sum += reward
        if reward > 0:
            print(f"You won {reward}")
        elif reward < 0:
            print(f"You lose {reward}")

        if done:
            print(f"The game has finished - total reward {reward_sum}")
            break

        state = state.unsqueeze(0).to(device)
        all_states = torch.cat([all_states, state], dim=0)

        rtgs += [rtgs[-1] - reward]

        # get last step_size size in sequence
        for t in range(time_start, len(all_states), step_size):
            batch_states = all_states.unsqueeze(0)[:, t:t+step_size]
            batch_actions = torch.tensor(actions+[0], dtype=torch.long).to(device).unsqueeze(0)[:, t:t+step_size]
            batch_rtgs = torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1)[:, t:t+step_size].type(torch.float32)
        time_start = t

        # pick up next action
        rtgs_emb, states_emb, actions_emb = emb(
            rtgs=batch_rtgs,
            states=batch_states,
            actions=batch_actions,
        )
        logits = gpt(
            rtgs_emb=rtgs_emb,
            states_emb=states_emb,
            actions_emb=actions_emb,
            timesteps=torch.arange(min(t, max_timesteps), min(t, max_timesteps) + len(states_emb)).unsqueeze(0).to(device),
        )
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        sampled_action = torch.multinomial(probs, num_samples=1)

# run_game(20.0, emb, gpt, device, stack_size, max_timesteps)

optimizer = gpt.configure_optimizers()

batch_size = 128
epochs = 1

# get state array (step_size+stack_size-1) h, w) by file idx
def get_state_from_index(dest_dir, step_size, stack_size, idx):
    img = Image.open(f"{dest_dir}/{idx}.png")
    arr = np.array(img)
    arr = arr.reshape(step_size + stack_size - 1, 84, 84)
    return arr


# array of single frame (batch, step_size+stack_size-1, h, w)
#     to frame stacking (batch, step_size, stack_size, h, w)
def toFrameStack(states, stack_size):
    new_states = states
    new_states = [np.roll(new_states, i, axis=1) for i in reversed(range(stack_size))]
    new_states = np.stack(new_states, axis=2)
    new_states = new_states[:, stack_size - 1:, :, :]
    return new_states


class AtariPongDataset(torch.utils.data.IterableDataset):
    def __init__(self, files_dir, batch_size, step_size, stack_size):
        super(AtariPongDataset).__init__()

        self.files_dir = files_dir
        self.step_size = step_size
        self.batch_size = batch_size
        self.stack_size = stack_size

        self.max_chunk = 1000

    def __len__(self):
        length = 0
        for chunk_num in range(self.max_chunk):
            # file check
            if not os.path.exists(f"{self.files_dir}/meta{chunk_num}.pkl"):
                break
            # read chunk
            with open(f"{self.files_dir}/meta{chunk_num}.pkl", "rb") as f:
                chunk = pickle.load(f)
            states_meta = chunk["states_meta"]
            # add batch count
            length += math.ceil(len(states_meta) / self.batch_size)
        return length

    def __iter__(self):
        for chunk_num in range(self.max_chunk):
            # file check
            if not os.path.exists(f"{self.files_dir}/meta{chunk_num}.pkl"):
                break
            # read chunk
            with open(f"{self.files_dir}/meta{chunk_num}.pkl", "rb") as f:
                chunk = pickle.load(f)
            rtgs = chunk["rtgs"]
            states_meta = chunk["states_meta"]
            actions = chunk["actions"]
            timesteps = chunk["timesteps"]
            # shuffle in chunk
            c = np.c_[
                rtgs.reshape(len(rtgs), -1),
                states_meta.reshape(len(states_meta), -1),
                actions.reshape(len(actions), -1),
                timesteps.reshape(len(timesteps), -1),
            ]
            np.random.shuffle(c)
            col = 0
            rtgs = c[:, col:col + self.step_size].reshape(rtgs.shape)
            col += self.step_size
            states_meta = c[:, col]
            col += 1
            actions = c[:, col:col + self.step_size].reshape(actions.shape)
            col += self.step_size
            timesteps = c[:, col:col + self.step_size].reshape(timesteps.shape)
            # process batch
            for i in range(0, len(states_meta), self.batch_size):
                # get rtgs, actions, timesteps
                rtgs_batch = rtgs[i:i + self.batch_size, :]
                actions_batch = actions[i:i + self.batch_size, :]
                timesteps_batch = timesteps[i:i + self.batch_size, :]
                # get states
                states_batch = [get_state_from_index(self.files_dir, self.step_size, self.stack_size, j) for j in
                                states_meta[i:i + self.batch_size]]
                states_batch = np.stack(states_batch, axis=0)
                states_batch = toFrameStack(states_batch, self.stack_size)
                # transform
                rtgs_batch = np.expand_dims(rtgs_batch.astype(float), axis=-1)
                states_batch = states_batch.astype(float) / 255.0
                # yield return
                yield torch.tensor(rtgs_batch, dtype=torch.float32).to(device), torch.tensor(states_batch,
                                                                                             dtype=torch.float32).to(
                    device), torch.tensor(actions_batch, dtype=torch.int64).to(device), torch.tensor(timesteps_batch,
                                                                                                     dtype=torch.int64).to(
                    device)

# load dataset
dataset = AtariPongDataset(
    files_dir=dest_dir,
    batch_size=batch_size,
    step_size=step_size,
    stack_size=stack_size,
)

# get total number of batch
total_batch = len(dataset)

# set cosine annealing cycle
lr_tokens_cycle = total_batch * batch_size * step_size // 20
processed_tokens = 0

# run training loop
for epoch_num in range(epochs):
    for i, (rtgs, states, actions, timesteps) in enumerate(dataset):
        # print("1")
        # create label (see above note)
        mask = (timesteps[:,1:]==0).int()
        mask = torch.cat((torch.zeros((mask.shape[0], 1), dtype=torch.int64).to(device), mask), dim=1)
        mask = torch.cumsum(mask, dim=1)
        labels = actions.masked_fill(mask != 0, -100)
        # process step
        optimizer.zero_grad()
        rtgs_emb, states_emb, actions_emb = emb(rtgs, states, actions)
        logits = gpt(rtgs_emb, states_emb, actions_emb, timesteps)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        # print loss
        print("epoch: "+str(epoch_num+1)+"   iter: "+str((i+1)/total_batch)+"   loss: "+str(loss.item()))
        train_loss.append(loss.item())
        # print(f"epoch {epoch_num+1} iter {i + 1}/{total_batch} - loss: {loss.item() :2.4f}", end="\r")
        # schedule learning rate (see above)
        processed_tokens += (labels >= 0).sum()
        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(processed_tokens) / float(lr_tokens_cycle))))
        lr = 6e-4 * lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    print("")
with open("./train_loss.txt", 'w') as train_los:
    train_los.write(str(train_loss))

print("done")

# save model
torch.save(emb.state_dict(), "model_embedding_rwkv")
torch.save(gpt.state_dict(), "model_gpt_rwkv")



# load model
emb = Embeddings_for_Atari(rwkv_config_1).to(device)
gpt = decisionRWKV(rwkv_config_1.vocab_size).to(device)
emb.load_state_dict(torch.load("model_embedding_rwkv"))
emb.eval()
gpt.load_state_dict(torch.load("model_gpt_rwkv"))
gpt.eval()

run_game(20.0, emb, gpt, device, stack_size, max_timesteps)