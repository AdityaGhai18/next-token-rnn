import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import CharTokenizer
from pathlib import Path

class TextDataset(Dataset):
    def __init__(self, data_dir="data", block_size=512):
        self.tokenizer = CharTokenizer()
        self.block_size = block_size
        self.sequences = []
        
        for path in sorted(Path(data_dir).glob("*.txt")):
            with open(path, "r", encoding="utf-8") as f:
                tokens = self.tokenizer.encode(f.read())
            for i in range(len(tokens) - block_size):
                self.sequences.append(tokens[i:i+block_size])
        
        self.sequences = torch.stack(self.sequences)
        print(f"{len(self.sequences):,} sequences ({self.sequences.nelement()*4/1e6:.1f}MB)")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1] #t0, t1, t2
        y = seq[1:]  #t1, t2, t3
        return x, y

    def debug_sample(self, num_samples=5):
        print("Inspecting dataset samples...")
        for i in range(min(num_samples, len(self))):
            x, y = self[i]
            print(f"Sample {i}:")
            print(f"  Input (x): {x}")
            print(f"  Target (y): {y}")
            print()

if __name__ == "__main__":
    dataset = TextDataset(block_size=512)
    dataset.debug_sample(num_samples=5)  # Debugging: Print 5 samples

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    x, y = next(iter(dataloader))
    print("Batch sample:")
    print(f"  Input batch (x): {x}")
    print(f"  Target batch (y): {y}")
