
import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  # noqa

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

from torch.utils.cpp_extension import load
import types
import math
T_MAX = 1024
wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=
                ['-res-usage', '--use_fast_math', '-O3','--maxrregcount=60', '-Xptxas=-O3', f'-DTmax={T_MAX}'])

rwkv_config_1 = types.SimpleNamespace()
rwkv_config_1.datafile = os.path.join("data", "enwik8")
rwkv_config_1.batch_size = 1
rwkv_config_1.ctx_len = 1024
rwkv_config_1.lr = 0.00001
rwkv_config_1.betas = (0.9, 0.999)
rwkv_config_1.eps = 1e-8
rwkv_config_1.block_size = 150
rwkv_config_1.device = "cuda" if torch.cuda.is_available() else "cpu"
rwkv_config_1.embd_pdrop = 0.1
rwkv_config_1.n_embd = 512
rwkv_config_1.n_layer = 12
rwkv_config_1.model_name = "rwkv_demo"
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

@dataclass
class TrainConfig:
    # wandb params
    project: str = "D-RWKV"
    group: str = "Gym-MuJoCo"
    name: str = "halfcheetah-medium-v2"
    # model params
    embedding_dim: int = rwkv_config_1.n_embd
    num_layers: int = rwkv_config_1.n_layer
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # training params
    env_name: str = "halfcheetah-medium-v2"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4
    # evaluation params
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_episodes: int = 100
    eval_every: int = 10_000
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_d4rl_trajectories(
    env_name: str, gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(env_name).get_dataset()
    traj, traj_len = [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class SequenceDataset(IterableDataset):
    def __init__(self, env_name: str, seq_len: int = 10, reward_scale: float = 1.0):
        self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)

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

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 1024,
        episode_len: int = 1000,
        embedding_dim: int = rwkv_config_1.n_embd,
        num_layers: int = rwkv_config_1.n_layer,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList([Block(action_dim) for _ in range(rwkv_config_1.n_layer)])

        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)
        for block in self.blocks:
            out = block(out)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 1::3]) * self.max_action

        return out


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    target_return: float,
    device: str = "cpu",
) -> Tuple[float, float]:
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )

    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)

        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )

        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    wandb_init(asdict(config))

    # data & dataloader setup
    dataset = SequenceDataset(
        config.env_name, seq_len=config.seq_len, reward_scale=config.reward_scale
    )
    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env=gym.make(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    # model & optimizer & scheduler setup
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )
    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]

        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=step,
        )

        # validation in the env for the actual online performance
        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed)
                eval_returns = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, eval_len = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                    )
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)

                normalized_scores = (
                    eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )
                wandb.log(
                    {
                        f"eval/{target_return}_return_mean": np.mean(eval_returns),
                        f"eval/{target_return}_return_std": np.std(eval_returns),
                        f"eval/{target_return}_normalized_score_mean": np.mean(
                            normalized_scores
                        ),
                        f"eval/{target_return}_normalized_score_std": np.std(
                            normalized_scores
                        ),
                    },
                    step=step,
                )
            model.train()

    if config.checkpoints_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))


if __name__ == "__main__":
    train()