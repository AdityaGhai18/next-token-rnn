import torch

class CharTokenizer:
    def __init__(self):
        self.vocab = {}

        self.characters = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ,.:;!?'-"/\\()[]{}|\n\t•–º◦©°ä—øãñ½¼²▪−√¥£ß´ª¾™ﬁõ►□′¨³≈ˆ§‰●ﬂ∆℡ƒð¡¦#$%&*+<=>@_`ìíî„♦✓�"""
        for index, char in enumerate(self.characters):
            self.vocab[char] = index


    def encode(self, input: str) -> torch.Tensor:
        tokens = []
        for char in input:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                print(f"{char} is missing, we have a problem fuck")
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, input: torch.Tensor) -> str:
        reverse = {v: k for k, v in self.vocab.items()}
        return "".join(reverse.get(token, "??") for token in input)

    def coverage_check(self, data_dir = "data"):
        from pathlib import Path
        data_dir = Path(data_dir)
        total_missing = set()
        for path in sorted(data_dir.glob("*.txt")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        # Find missing chars in this file
            file_chars = set(text)
            vocab_chars = set(self.vocab.keys())
            missing = sorted(file_chars - vocab_chars)
            if missing:
                print(f"❌ {path.name}: {missing}")
                total_missing.update(missing)
            else:
                print(f"✅ {path.name}")
    
        if total_missing:
            print(f"\n🚨 ADD THESE TO self.characters: '{''.join(sorted(total_missing))}'")
        else:
            print("\n🎉 ALL GOOD!")

tokenizer = CharTokenizer()
tokenizer.coverage_check()



