import torch
from model_example import TextGenerationModel
from utils import load_model

# Preparación de datos
text = "hola como estas hola bien y tu hola genial"  
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

seq_length = 10

# Usar los mismos parámetros que en el entrenamiento
embedding_dim = 64
hidden_dim = 128
num_layers = 2
dropout = 0.3

model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
model = load_model(model, "best_model.pth")  # Cargar el mejor modelo

def generate_text(model, start_string, length=100, temperature=1.0):
    model.eval()
    chars = [char_to_idx.get(ch, char_to_idx[' ']) for ch in start_string]
    while len(chars) < seq_length:
        chars = [char_to_idx[' ']] + chars
    
    hidden = model.init_hidden(1)
    generated_chars = list(chars)
    
    print(f"Vocabulary size: {vocab_size}")
    
    for _ in range(length):
        x = torch.tensor([chars[-seq_length:]], dtype=torch.long)
        output, hidden = model(x, hidden)
        
        # Debug información
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        # Usar torch.multinomial
        probs = torch.softmax(output.squeeze(), dim=-1)
        print(f"Probabilities shape: {probs.shape}")
        
        next_char = torch.multinomial(probs, 1).item()
        generated_chars.append(next_char)
        chars = chars[1:] + [next_char]
    
    return ''.join([idx_to_char[i] for i in generated_chars])

start_string = "hola"
generated_text = generate_text(model, start_string, length=50, temperature=0.8)
print("Texto generado:", generated_text)
