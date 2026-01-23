import sys
import torch
from train import VanillaRNN, generate, CharTokenizer

if len(sys.argv) != 3:
    print("Usage: python3 infer.py <checkpoint.pt> <prompt>")
    sys.exit(1)
                            
checkpoint_path, prompt = sys.argv[1], sys.argv[2]
                                    
tokenizer = CharTokenizer()
vocab_size = len(tokenizer.vocab)
                                                
model = VanillaRNN(vocab_size).cuda()
model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
model.eval()
                                                                
print(f"Loaded {checkpoint_path}")
output = generate(model, tokenizer, prompt, max_new_tokens=2000)
print(f"\nPrompt: {prompt}")
print(f"Generated:\n{output}")

