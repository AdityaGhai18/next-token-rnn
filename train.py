import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import TextDataset
from tokenizer import CharTokenizer
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, block_size=512, embed_dim=64, hidden_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embedding and linear layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.W_xh = nn.Linear(embed_dim, hidden_dim)    # input → hidden
        self.W_hh = nn.Linear(hidden_dim, hidden_dim)   # hidden → hidden
        self.W_ho = nn.Linear(hidden_dim, vocab_size)   # hidden → output
        self.h_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, T = x.shape
        x = self.token_embedding(x)  # (B, T, embed)
        h = torch.zeros(B, self.hidden_dim, device=x.device)  # initial hidden
        logits = torch.zeros(B, T, self.vocab_size, device=x.device)

        for t in range(T):
            x_t = x[:, t]
            h = torch.tanh(self.W_xh(x_t) + self.W_hh(h))
            h = self.h_norm(h)
            logits[:, t] = self.W_ho(h)
        return logits  # (B, T, vocab_size)

    def step(self, x_t, h):
        x_t = self.token_embedding(x_t)
        h = torch.tanh(self.W_xh(x_t) + self.W_hh(h))
        h = self.h_norm(h)
        logits = self.W_ho(h)
        return logits, h

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.1):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, device='cuda').unsqueeze(0)
    B = 1
    
    with torch.no_grad():
        h = torch.zeros(B, model.hidden_dim, device='cuda')
        for t in range(x.size(1)):
            _, h = model.step(x[:, t], h) #building hidden state from prompt
    
    cur_token = x[:, -1] #last token


    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, h = model.step(cur_token, h)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            #next_token = torch.argmax(probs, dim = -1)


        x = torch.cat([x, next_token.unsqueeze(1)], dim=1)
        cur_token = next_token

    return tokenizer.decode(x[0].tolist())

def train(
    data_dir="data",
    block_size=512,
    batch_size=64,
    embed_dim=64,
    hidden_dim=512,
    lr=1e-4,  
    epochs=50,
    checkpoint_dir="checkpoints_run3",
    resume_from=None
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = CharTokenizer()
    vocab_size = len(tokenizer.vocab)

    model = VanillaRNN(vocab_size, block_size, embed_dim, hidden_dim).cuda()

    # Apply Xavier initialization to weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)

    dataset = TextDataset(data_dir=data_dir, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    start_epoch = 1  # default starting epoch

    if resume_from is not None and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location='cuda')
        model.load_state_dict(checkpoint)
        # Extract epoch number from filename
        import re
        m = re.search(r"rnn_epoch_(\d+).pt", resume_from)
        if m:
            start_epoch = int(m.group(1)) + 1

        print(f"Resuming training from {resume_from} at epoch {start_epoch}")


    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Dataset size: {len(dataset):,} sequences")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0.0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (x, y) in pbar:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x)

            if batch_idx % 100 == 0:
                # Debugging: Check shapes
                print(f"Batch {batch_idx}:")
                print(f"  Input shape: {x.shape}")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Target shape: {y.shape}")

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if batch_idx % 100 == 0:
                # Debugging: Check loss
                print(f"  Loss: {loss.item()}")
                # Debugging: Check gradient norms
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
                print(f"  Gradient norm: {grad_norm}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        epoch_time = (time.time() - epoch_start_time) / 60
        print(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f} min")
        
                
        checkpoint_path = os.path.join(checkpoint_dir, f"rnn_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"SAVED {checkpoint_path}")



        # Generate a small sample
        prompt = "INT. ROOM - NIGHT\nAdi sits quietly at the table waiting for Luke Skywalker"
        sample = generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8)
        print(f"Generated text sample:\n{sample}\n{'-'*80}")

if __name__ == "__main__":
    train(
        block_size=512,
        batch_size=64,      
        epochs=50,
    )

