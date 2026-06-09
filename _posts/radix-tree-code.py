import torch
import torch.nn as nn
from torch.nn import functional as F
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# # hyperparameters
# batch_size = 64 # how many independent sequences will we process in parallel?
# block_size = 256 # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2
# ------------

# hyperparameters for testing

batch_size = 8          # smaller training batches
block_size = 64        # keep same for now so your benchmark assumptions hold
max_iters = 120         # much faster than 5000
eval_interval = 20
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10         # much faster validation
n_embd = 32             # was 64
n_head = 4              # 32 / 4 = 8 dim per head
n_layer = 4             # was 4
dropout = 0.0

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss, _ = model(X, Y)  # unpack 3 return values now
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



def load_from_radix_tree(request, tree, prompt_tokens, block_size):
    """Load cached KV from the radix tree onto a request."""
    node, matched = tree.match_prefix(prompt_tokens)

    if matched == 0: return 0

    prefix_path = []
    curr = node

    while curr != tree.root:
        prefix_path.append(curr)
        curr = curr.parent
    
    prefix_path.reverse()

    for (layer, head) in prefix_path[0].kv_data.keys():
        new_k, new_v = [], []

        for pnode in prefix_path:
            pk, pv = pnode.kv_data[(layer, head)]
            new_k.append(pk.clone())
            new_v.append(pv.clone())
            
        request.kv_cache[(layer, head)] = (
            torch.cat(new_k, dim=1),
            torch.cat(new_v, dim=1),
        )

    for pnode in prefix_path:
        pnode.lock_ref += 1        
    
    # Snap to block boundary for prefill_cursor
    num_cached = (matched // block_size) * block_size
    request.prefill_cursor = num_cached
    request._radix_path = prefix_path  # save for later unlock
    return num_cached

class RadixNode:
    def __init__(self):
        self.children: Dict[int, RadixNode] = {}
        self.parent: Optional[RadixNode] = None
        self.token_ids: Tuple[int, ...] = ()
        self.kv_data: Optional[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]] = None
        self.lock_ref: int = 0
        self.last_access_time: int = 0


class RadixTree:
    def __init__(self):
        self.root = RadixNode()
        self.step = 0
    
    def pretty_print(self, node=None, indent=0):
        if node is None:
            node = self.root
            print("RadixTree:")
        
        prefix = "  " * indent
        token_str = str(list(node.token_ids)[:8])
        if len(node.token_ids) > 8:
            token_str += "..."
        kv_str = f"KV[{node.kv_data is not None}]" if node != self.root else "ROOT"
        lock_str = f"lock={node.lock_ref}"
        print(f"{prefix}{token_str} ({kv_str}, {lock_str}, t={node.last_access_time})")
        
        for child in node.children.values():
            self.pretty_print(child, indent + 1)
    
    def _split_node(self, child: RadixNode, split_len: int) -> RadixNode:
        """Split child's edge at position split_len. Returns the new mid-node."""
        new_mid = RadixNode()
        new_mid.token_ids = child.token_ids[:split_len]
        new_mid.parent = child.parent
        new_mid.last_access_time = child.last_access_time
        new_mid.lock_ref = child.lock_ref

        if child.kv_data is not None:
            new_mid.kv_data = {}
            new_child_kv = {}

            for (layer, head), (k, v) in child.kv_data.items():
                new_mid.kv_data[(layer, head)] = (
                    k[:, :split_len, :].clone(),
                    v[:, :split_len, :].clone()
                )
                new_child_kv[(layer, head)] = (
                    k[:, split_len:, :].clone(),
                    v[:, split_len:, :].clone()
                )
            
            child.kv_data = new_child_kv

        suffix_tokens = child.token_ids[split_len:]
        child.token_ids = suffix_tokens
        child.parent = new_mid

        new_mid.children[suffix_tokens[0]] = child

        new_mid.parent.children[new_mid.token_ids[0]] = new_mid
        
        return new_mid
    
    def insert(self, token_ids: List[int], kv_data_full: Dict, block_size: int):
        """Insert token_ids and their KV data into the tree."""

        node, matched = self.match_prefix(token_ids)

        if matched == len(token_ids): return

        remaining = token_ids[matched:]
        new_node = RadixNode()
        new_node.token_ids = tuple(remaining)
        new_node.parent = node
        new_node.last_access_time = self.step

        new_node.kv_data = {}


        for (layer, head), (k, v) in kv_data_full.items():
            new_node.kv_data[(layer, head)] = (
                k[:, matched:matched + len(remaining)].clone(),
                v[:, matched:matched + len(remaining)].clone()
            )
        
        node.children[remaining[0]] = new_node
        
    def _find_leaves(self, node, result):
        if not node.children:
            result.append(node)
        for child in node.children.values():
            self._find_leaves(child, result)

    def unlock_radix_path(self, request):
        """Release the tree locks acquired during load_from_radix_tree."""
        path = getattr(request, '_radix_path', None)
        if path is None:
            return
        for node in path:
            node.lock_ref -= 1
        request._radix_path = None

    def match_prefix(self, token_ids: List[int]) -> Tuple[RadixNode, int]:
        """
        Find the longest prefix of token_ids that exists in the tree.
        Returns (last_matched_node, num_matched_tokens).
        
        IMPORTANT: If the match ends in the MIDDLE of an edge, you must
        SPLIT the edge so there's a node at the exact match boundary.
        """
        node = self.root
        matched = 0

        while matched < len(token_ids):
            next_token = token_ids[matched]
            child = node.children.get(next_token)

            if child is None: break

            edge_tokens = child.token_ids
            edge_match_len = 0

            while (edge_match_len < len(edge_tokens) and 
                matched + edge_match_len < len(token_ids) and 
                edge_tokens[edge_match_len] == token_ids[matched + edge_match_len]):
                edge_match_len += 1

            if edge_match_len < len(edge_tokens):
                child = self._split_node(child, edge_match_len)
                matched += edge_match_len
                node = child
                break

            matched += edge_match_len
            node = child

        return node, matched

@dataclass
class Request:
    """Each in-flight generation carries its own state and KV cache."""
    id: int
    prompt_tokens: List[int]          # the original encoded prompt
    max_new_tokens: int               # how many tokens this request wants
    generated_tokens: List[int] = field(default_factory=list)
    status: str = "waiting"           # "waiting" -> "prefilling" -> "active" -> "done"
    prefill_cursor: int = 0
    _radix_path: list = field(default_factory=list)

    # Hint 2: Per-request KV cache, keyed by (layer_idx, head_idx)
    # Each value is a (key_tensor, value_tensor) tuple of shape (1, T_i, head_size)
    # T_i grows by 1 each decode step — different requests have different T_i
    kv_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    @property
    def tokens_so_far(self) -> List[int]:
        """Full sequence: prompt + everything generated."""
        return self.prompt_tokens + self.generated_tokens

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def is_done(self) -> bool:
        return self.num_generated >= self.max_new_tokens
    
    @property
    def is_fully_prefilled(self) -> bool:
        return self.prefill_cursor == len(self.prompt_tokens)

    def clear_cache(self):
        self.kv_cache.clear()

class Scheduler:
    def __init__(self, policy="fcfs", max_batch_size=4, token_budget=16, max_kv_tokens=22, block_size=4):
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.token_budget = token_budget
        self.max_kv_tokens = max_kv_tokens
        self.block_size = block_size
        self.radix_tree = RadixTree()

        self.waiting = []
        self.prefilling = []
        self.active = []
        self.preempted = []

    def promote(self, req):
        self.prefilling.remove(req)
        req.status = "active"
        self.active.append(req)
    
    def complete(self, req):
        self.active.remove(req)
        req.status = "done"
    
    def _sort_key(self, req):
        if self.policy == "fcfs":
            return (0, req.arrival_time)

    def add_request(self, req):
        key = self._sort_key(req)
        heapq.heappush(self.waiting, (*key, req.id, req))
    
    def is_done(self):
        return not (self.waiting or self.prefilling or self.active)
    
    def _maybe_admit(self, step):
        if self.prefilling:
            return
        
        if not self.waiting:
            return

        kv_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active + self.prefilling)

        _, _, _, candidate = self.waiting[0]

        _, num_cached = self.radix_tree.match_prefix(candidate.prompt_tokens)        
        num_cached = (num_cached // self.block_size) * self.block_size
        actual_kv_cost = len(candidate.prompt_tokens) - num_cached

        if kv_used + actual_kv_cost > self.max_kv_tokens:
            return
        
        if len(self.active) + len(self.prefilling) >= self.max_batch_size: return

        heapq.heappop(self.waiting)
        load_from_radix_tree(candidate, self.radix_tree, candidate.prompt_tokens, self.block_size)
        candidate.arrival_time = step
        candidate.status = "prefilling"
        self.prefilling.append(candidate)
    
    def _maybe_preempt(self):
        kv_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active + self.prefilling)

        while self.active and kv_used > self.max_kv_tokens:
            victim = max(self.active, key=lambda r: (r.priority, -r.arrival_time))
            self.active.remove(victim)
            victim.clear_cache()
            victim.prefill_cursor = 0
            victim.status = "waiting"
            self.preempted.append(victim)

            key = self._sort_key(victim)
            heapq.heappush(self.waiting, (*key, victim.id, victim))
            kv_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active + self.prefilling)

    def schedule(self, step: int):
        """
        Returns:
            prefill_req:  Request | None  — one request getting a prefill chunk (or None)
            decode_reqs:  List[Request]   — all requests currently being decoded (active)

        """
        self.radix_tree.step = step

        self._maybe_admit(step)       # promote waiting → prefilling if memory allows
        self._maybe_preempt()         # evict if over memory budget

        prefill_req = self.prefilling[0] if self.prefilling else None
        decode_reqs = list(self.active)

        return prefill_req, decode_reqs


class Head(nn.Module):
    """One head of self-attention — now STATELESS (no internal cache)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_k=None, past_v=None, attn_mask=None):
        """
        Args:
            x:      (B, T, C)       input embeddings
            past_k: (B, T_past, hs) cached keys, or None
            past_v: (B, T_past, hs) cached values, or None
        Returns:
            out:   (B, T, hs)           attention output
            new_k: (B, T_past+T, hs)    updated key cache   (None during training)
            new_v: (B, T_past+T, hs)    updated value cache  (None during training)
        """
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        if not self.training:
            if past_k is not None:
                T_past = past_k.shape[1]
                # ── Decode step: append new K/V onto cached past ──
                k = torch.cat([past_k, k], dim=1)  # (B, T_past + T, hs)
                v = torch.cat([past_v, v], dim=1)

                # Q attends over full cache — no causal mask needed (T=1)
                wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

            # ── Causal mask for multi-token prefill continuation ──
            # When T > 1, new tokens must not attend to future tokens
            # within the chunk. (When T == 1, this is a no-op — skip it.)                

                if T > 1:
                    past_part = torch.ones(T, T_past, device=wei.device, dtype=torch.bool)
                    new_part = torch.tril(torch.ones(T, T, device=wei.device, dtype=torch.bool))
                    causal_mask = torch.cat([past_part, new_part], dim=-1)
                    wei = wei.masked_fill(~causal_mask, float('-inf'))

                if attn_mask is not None:
                    new_valid = torch.ones(B, 1, T, device=wei.device, dtype=torch.bool)
                    full_mask = torch.cat([attn_mask, new_valid], dim=-1)
                    wei = wei.masked_fill(~full_mask, float('-inf'))

                wei = F.softmax(wei, dim=-1)
                wei = self.dropout(wei)
                out = wei @ v
            else:
                # ── Prefill step: full prompt, needs causal mask ──
                wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
                wei = F.softmax(wei, dim=-1)
                wei = self.dropout(wei)
                out = wei @ v

            return out, k, v   # return updated cache
        else:
            # ── Training path — unchanged, no cache ──
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out, None, None


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, attn_mask=None):
        """
        Args:
            x:       (B, T, C)
            past_kv: list of (past_k, past_v) per head, or None
        Returns:
            out:    (B, T, n_embd)
            new_kv: list of (new_k, new_v) per head
        """
        if past_kv is None:
            past_kv = [(None, None)] * len(self.heads)

        outputs, new_kvs = [], []
        for i, h in enumerate(self.heads):
            pk, pv = past_kv[i]
            out, nk, nv = h(x, pk, pv, attn_mask=attn_mask)
            outputs.append(out)
            new_kvs.append((nk, nv))

        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, new_kvs


class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, past_kv=None, attn_mask=None):
        """
        Returns:
            x:      (B, T, n_embd)
            new_kv: list of (new_k, new_v) per head in this block
        """
        sa_out, new_kv = self.sa(self.ln1(x), past_kv, attn_mask=attn_mask)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # ModuleList instead of Sequential so we can pass per-block cache
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pos=None, past_kvs=None, attn_mask=None):
        """
        Args:
            idx:      (B, T) token indices
            targets:  (B, T) target indices, or None
            pos:      (B, T) explicit position indices, or None (uses arange)
            past_kvs: list-of-lists cache structure, or None
                      past_kvs[layer][head] = (key_tensor, value_tensor)
        Returns:
            logits:   (B, T, vocab_size)
            loss:     scalar or None
            new_kvs:  updated cache with same structure as past_kvs
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)

        if pos is None:
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        else:
            pos_emb = self.position_embedding_table(pos)  # (B, T, C)

        x = tok_emb + pos_emb  # (B, T, C)

        # Thread cache through each block
        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, block_kv = block(x, past_kvs[i], attn_mask=attn_mask)
            new_kvs.append(block_kv)

        x = self.ln_f(x)          # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_kvs

    def generate(self, idx, max_new_tokens):
        """Original generate (no cache, full recompute) for reference."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss, _ = model(xb, yb)  # _ discards the cache during training
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Quick sanity check with the original no-cache generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

def assemble_batch_cache(requests):
    """
    Gather per-request KV caches into batched tensors.
    LEFT-pads shorter caches so new tokens always land at the right edge.

    Big problem: You have 3 active requests. Each owns its own KV cache. You need to feed them to the model as one
    batched tensor. But their caches have different lengths:

    Returns:
        past_kvs:    batched cache structure  [layer][head] = (B, T_max, hs)
        attn_mask:   (B, 1, T_max) bool — True = valid, False = padding
        pad_lengths: list of int — how many pad positions per request (for disassembly)
    """

    B = len(requests)
    lengths = [req.kv_cache[(0, 0)][0].shape[1] for req in requests]
    max_t = max(lengths)

    pad_lengths = [max_t - t for t in lengths] # pad lengths for every position in t

    attn_mask = torch.zeros(B, 1, max_t, device=device, dtype=torch.bool)

    for i, pad in enumerate(pad_lengths):
        attn_mask[i, 0, pad:] = True

    past_kvs = []

    for layer_idx in range(n_layer):
        block_kv = []

        for head_idx in range(n_head):
            keys, values = [], []

            for i, req in enumerate(requests):
                k, v = req.kv_cache[(layer_idx, head_idx)]
                if pad_lengths[i] > 0:
                    hs = k.shape[2]
                    pad = torch.zeros(1, pad_lengths[i], hs, device=device)
                    k = torch.cat([pad, k], dim=1)
                    v = torch.cat([pad, v], dim=1)

                keys.append(k)
                values.append(v)

            block_kv.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        past_kvs.append(block_kv)

    return past_kvs, attn_mask, pad_lengths

def disassemble_batch_cache(requests, new_kvs, pad_lengths):
    """
    Scatter batched KV cache back to per-request storage.
    After Head's torch.cat, each row is (T_max + 1) — strip the left-padding.
    """
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(requests):
                pad = pad_lengths[i]
                req.kv_cache[(layer_idx, head_idx)] = (
                    batched_k[i : i + 1, pad:, :],      # (1, T_i + 1, hs)
                    batched_v[i : i + 1, pad:, :],
                )

def insert_into_radix_tree(request, tree, block_size):
    """Insert a request's prompt tokens and KV data into the radix tree."""
    tree.insert(request.prompt_tokens, request.kv_cache, block_size)

def scheduled_generate(model, requests, policy="fcfs", token_budget=16, max_kv_tokens=256):
    scheduler = Scheduler(policy, token_budget=token_budget, max_kv_tokens=max_kv_tokens)

    step = 0

    for req in requests:
        req.arrival_time = step
        scheduler.add_request(req)
    
    model.eval()

    with torch.no_grad():
        while not scheduler.is_done():

            prefill_req, decode_reqs = scheduler.schedule(step)

            if prefill_req:

                prefill_chunk_tokens = []

                remaining_budget = token_budget - len(scheduler.active)

                if remaining_budget > 0 and scheduler.prefilling:
                    p_req = scheduler.prefilling[0]

                    tokens_left = len(p_req.prompt_tokens) - p_req.prefill_cursor
                    chunk_size = min(remaining_budget, tokens_left)

                    chunk_start = p_req.prefill_cursor 

                    chunk_tokens = p_req.prompt_tokens[chunk_start: chunk_start + chunk_size]

                    prefill_chunk_tokens = torch.tensor([chunk_tokens], dtype=torch.long, device=device)

                    p_req.prefill_cursor += chunk_size

                if len(prefill_chunk_tokens) == 0 and not scheduler.active:
                    step += 1
                    continue

                if len(prefill_chunk_tokens) > 0:
                    pos = torch.arange(chunk_start, chunk_start + chunk_size, device=device).unsqueeze(0)

                    if p_req.kv_cache:
                        past_kvs = []
                        for layer_idx in range(n_layer):
                            block_kv = [(p_req.kv_cache[(layer_idx, hi)]) for hi in range(n_head)] 
                            past_kvs.append(block_kv)
                        
                        logits, _, new_kvs = model(prefill_chunk_tokens, pos=pos, past_kvs=past_kvs)

                    else:
                        logits, _, new_kvs = model(prefill_chunk_tokens, pos=pos)

                    for li, bkv in enumerate(new_kvs):
                        for hi, (k, v) in enumerate(bkv):
                            p_req.kv_cache[(li, hi)] = (k, v)
            
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                    if prefill_req.is_fully_prefilled:
                        insert_into_radix_tree(prefill_req, scheduler.radix_tree, scheduler.block_size)
                        prefill_req.generated_tokens.append(idx_next.item())
                        prefill_req._last_token = idx_next
                        scheduler.radix_tree.unlock_radix_path(prefill_req)
                        scheduler.promote(prefill_req)
    
            if decode_reqs:

                batch_tokens = torch.cat([req._last_token for req in scheduler.active])

                batch_positions = torch.tensor([[len(req.tokens_so_far) - 1] for req in scheduler.active], device=device)

                past_kvs, attn_mask, pad_lengths = assemble_batch_cache(scheduler.active)

                logits, _, new_kvs = model(
                    batch_tokens,
                    pos=batch_positions,
                    past_kvs=past_kvs,
                    attn_mask=attn_mask
                )

                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                disassemble_batch_cache(scheduler.active, new_kvs, pad_lengths)

                for i, req in enumerate(scheduler.active):
                    req.generated_tokens.append(idx_next[i].item())
                    req._last_token = idx_next[i : i + 1]
                
                for req in list(scheduler.active):
                    if req.is_done:
                        scheduler.radix_tree.unlock_radix_path(req)
                        scheduler.complete(req)
        
            step += 1
    
    return scheduler

# ── Run benchmarks ────────────────────────────────────────────────────────────

from benchmarks.radix_tree_benchmark_runs import (
    run_radix_tree_benchmark_suite,
)

run_radix_tree_benchmark_suite(
    m,
    vocab_size=vocab_size,
    device=device,
    block_size=block_size,
)