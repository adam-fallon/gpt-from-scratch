#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# hyperparams
batch_size = 32  # how many independent sequences will we process in parallel
block_size = 8  # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32
# ----------


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, n_embed), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, 16) @ (B, 16, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        # out = wei @ x
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed // 4)
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        # Logits are effectively scores for the likelehood of a token occuring
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = token_emb + pos_emb  # (B,T,C)
        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape the logits to work with cross entropy
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            # Reshape the targets to work with cross entropy
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get predictions
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # apply sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx


def main():
    with open("shakespeare.txt", "r") as f:
        text = f.read()
        print(f"Length of text: {len(text)} characters")

        # Get all unique characters
        chars = sorted(list(set(text)))
        # What is the length of the characters
        vocab_size = len(chars)
        print("".join(chars))
        print(vocab_size)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [
            stoi[c] for c in s
        ]  # encoder: take a string, output a list of integers
        decode = lambda l: "".join(
            [itos[i] for i in l]
        )  # decoder: take a list of integers, output a string

        data = torch.tensor(encode(text), dtype=torch.long)
        print(data.shape, data.dtype)
        # print(data[:1000]) # first 1000 characters

        # Split training and validation data
        n = int(0.9 * len(data))
        train_data = data[:n]  # 90% of data
        val_data = data[n:]  # 10% of data

        # Chunking
        train_data[: block_size + 1]

        x = train_data[:block_size]
        y = train_data[1 : block_size + 1]

        def get_batch(split):
            data = train_data if split == "train" else val_data
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[i : i + block_size] for i in ix])
            y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
            x, y = x.to(device), y.to(device)
            return x, y

        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ["train", "val"]:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = get_batch(split)
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out

        xb, yb = get_batch("train")
        print("inputs:")
        print(xb.shape)
        print(xb)
        print("targets:")
        print(yb.shape)
        print(yb)

        print("----")

        model = BigramLanguageModel(vocab_size)
        m = model.to(device)
        logits, loss = m(xb, yb)
        print(logits.shape)
        print(loss)

        # TRAIN

        # Create a PyTorch optimizer using a popular optimizer.
        # lr = learning rate
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']}, val loss {losses['val']}"
            )
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Loss rate after training: {loss.item()}")

    # Couple tokens together
    # Trick of self attention;
    #   torch.tril(torch.ones(3,3)) <- returns lower triangle part of
    #   [1,0,0]
    #   [1,1,0]
    #   [1,1,1]

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


main()
