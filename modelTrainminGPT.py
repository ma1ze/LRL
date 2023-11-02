import torch
import torch.nn as nn
from torch.nn import functional as F
import atari_py
import cv2
import matplotlib.pyplot as plt
import torch
import io
import math
import numpy as np
import os
import gzip
import pickle
from PIL import Image
import PIL
train_loss = []
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

ale = atari_py.ALEInterface()
ale.loadROM(atari_py.get_game_path("pong"))
ale.reset_game()
available_actions = ale.getMinimalActionSet()
action_dict = dict([i, e] for i, e in enumerate(available_actions))

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT_for_DT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.block_size = config.block_size
        self.n_embd = config.n_embd

        # build modules
        self.global_timestep_encoding = nn.Embedding(config.max_timestep, config.n_embd)
        self.context_position_encoding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.embd_pdrop)
        self.block_loop = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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

    def forward(self, rtgs_emb, states_emb, actions_emb, timesteps):
        # rtgs_emb    : (batch_size, step_size, n_embd)
        # states_emb  : (batch_size, step_size, n_embd)
        # actions_emb : (batch_size, step_size, n_embd)
        # timesteps   : (batch_size, step_size)  <-- but only the first step is used (other steps are ignored)

        batch_size = states_emb.shape[0]
        actual_step_size = states_emb.shape[1]

        #
        # Generate a sequence of tokens :
        # [s], [a], [R] --> [R, s, a, R, s, a, ...]
        #

        token_emb = torch.zeros(
            (batch_size, actual_step_size*3, self.n_embd),
            dtype=torch.float32,
            device=states_emb.device)
        token_emb[:,::3,:] = rtgs_emb
        token_emb[:,1::3,:] = states_emb
        if actions_emb is not None:
            token_emb[:,2::3,:] = actions_emb

        #
        # Position encoding
        #

        timestep_start = torch.repeat_interleave(timesteps[:,0].unsqueeze(dim=-1), actual_step_size*3, dim=-1) # (batch_size, actual_step_size*3)
        pos_global = self.global_timestep_encoding(timestep_start)
        context_position = torch.arange(actual_step_size*3, device=states_emb.device).repeat(batch_size,1) # (batch_size, actual_step_size*3)
        pos_relative = self.context_position_encoding(context_position)
        pos_emb = pos_global + pos_relative

        x = self.dropout(token_emb + pos_emb)

        #
        # Apply multi-layered MHA (multi-head attentions)
        #

        for block in self.block_loop:
            x = block(x)

        x = self.norm(x)

        #
        # Apply Feed-Forward and Return
        #

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
class CfgNode:
    n_head = 8
    n_layer = 6
    n_embd = 128  # each head has n_embd / n_head
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    block_size = step_size * 3
    max_timestep = max_timesteps
    vocab_size = len(action_dict)  # all actions

config = CfgNode()
gpt = GPT_for_DT(config).to(device)
emb = Embeddings_for_Atari(config).to(device)


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

def configure_optimizers(gpt, emb, weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95)):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in gpt.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    for mn, m in emb.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    gpt_param_dict = {pn: p for pn, p in gpt.named_parameters()}
    emb_param_dict = {pn: p for pn, p in emb.named_parameters()}
    param_dict = {**gpt_param_dict, **emb_param_dict}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer

optimizer = configure_optimizers(gpt, emb)

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
with open("./train_loss_transformer.txt", 'w') as train_los:
    train_los.write(str(train_loss))
print("done")

# save model
torch.save(emb.state_dict(), "model_embedding")
torch.save(gpt.state_dict(), "model_gpt")


# load model
emb = Embeddings_for_Atari(config).to(device)
gpt = GPT_for_DT(config).to(device)
emb.load_state_dict(torch.load("model_embedding"))
emb.eval()
gpt.load_state_dict(torch.load("model_gpt"))
gpt.eval()

run_game(20.0, emb, gpt, device, stack_size, max_timesteps)