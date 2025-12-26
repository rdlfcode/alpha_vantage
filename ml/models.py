"""
ML Models for AlgoEvals.
Includes Statistical Models, Vanilla Transformer, and TITANS.
"""
import torch
import torch.nn as nn
import math

# --- Statistical Models ---

class StatisticalModel(nn.Module):
    def __init__(self, pred_len=1):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x):
        raise NotImplementedError

class LinearRegressionModel(StatisticalModel):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch, seq_len, dim = x.shape
        x_flat = x.view(batch, -1)
        
        # Dynamic resizing for flexibility during experimentation
        if self.linear.in_features != seq_len * dim:
             device = x.device
             self.linear = nn.Linear(seq_len * dim, self.linear.out_features).to(device)
             
        return self.linear(x_flat)

class ExponentialRegressionModel(StatisticalModel):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Model: y = exp(Linear(x))
        batch, seq_len, dim = x.shape
        x_flat = x.view(batch, -1)
        
        if self.linear.in_features != seq_len * dim:
             device = x.device
             self.linear = nn.Linear(seq_len * dim, self.linear.out_features).to(device)

        return torch.exp(self.linear(x_flat))

class FFTModel(StatisticalModel):
    """
    FFT Model capable of operating on target only (1D) or all features (2D).
    """
    def __init__(self, seq_len, input_dim=1, pred_len=1, use_all_features=False):
        super().__init__(pred_len)
        self.seq_len = seq_len
        self.use_all_features = use_all_features
        
        if use_all_features:
            # 2D FFT on (seq, dim)
            # rfft2 output last dimension is dim // 2 + 1
            # We flatten the result: seq_len * (dim // 2 + 1)
            fft_dim = seq_len * (input_dim // 2 + 1)
        else:
            # 1D FFT on first feature
            fft_dim = seq_len // 2 + 1
            
        self.linear = nn.Linear(fft_dim, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        batch, seq, dim = x.shape
        
        if not self.use_all_features:
            # 1D FFT on the first feature
            x_signal = x[..., 0] 
            # rfft returns complex tensor of size floor(n/2) + 1
            fft_coeffs = torch.fft.rfft(x_signal, dim=-1)
            mags = torch.abs(fft_coeffs) # (batch, freq_bins)
        else:
            # 2D FFT on (seq, dim)
            # Computes 2D FFT over the last two dimensions
            fft_coeffs = torch.fft.rfft2(x)
            mags = torch.abs(fft_coeffs)
            mags = mags.view(batch, -1) # Flatten features
            
        return self.linear(mags)

# --- Transformer Models ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1).view(1, x.size(1), -1)
        return self.dropout(x)

class VanillaTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, output_dim=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src, src_mask=None):
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = output[:, -1, :]
        return self.decoder(output)

# --- TITANS Architecture ---

class MemoryModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x, state=None):
        raise NotImplementedError

class NeuralMemory(MemoryModule):
    """
    Titans Neural Memory: Updates weights dynamically.
    approximated via Hebbian-like update M_t = M_{t-1} + v_t * k_t^T
    """
    def __init__(self, d_model, memory_size=128):
        super().__init__(d_model)
        self.memory_size = memory_size
        self.key_proj = nn.Linear(d_model, memory_size)
        self.val_proj = nn.Linear(d_model, memory_size)
        self.query_proj = nn.Linear(d_model, memory_size)
        self.out_proj = nn.Linear(memory_size, d_model)

    def forward(self, x, state=None):
        batch, seq, _ = x.shape
        
        if state is None:
             state = torch.zeros(batch, self.memory_size, self.memory_size, device=x.device)
             
        outputs = []
        for t in range(seq):
            xt = x[:, t, :]
            k = self.key_proj(xt) # (b, mem)
            v = self.val_proj(xt) # (b, mem)
            q = self.query_proj(xt) # (b, mem)
            
            # Retrieve: M * q
            mem_out = torch.bmm(state, q.unsqueeze(2)).squeeze(2) # (b, m)
            
            # Update: M = decay * M + v * k^T
            update = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
            state = 0.99 * state + 0.01 * update 
            
            outputs.append(mem_out)
            
        outputs = torch.stack(outputs, dim=1) # (b, seq, mem)
        return self.out_proj(outputs), state

class LSTMMemory(MemoryModule):
    """
    LSTM-based Long Term Memory
    """
    def __init__(self, d_model, memory_size=128):
        super().__init__(d_model)
        # memory_size here implies hidden size of LSTM
        self.lstm = nn.LSTM(d_model, memory_size, batch_first=True)
        self.out_proj = nn.Linear(memory_size, d_model)
        
    def forward(self, x, state=None):
        # x: (b, seq, d_model)
        output, (hn, cn) = self.lstm(x, state)
        return self.out_proj(output), (hn, cn)

class TransformerMemory(MemoryModule):
    """
    Transformer-based Long Term Memory.
    Uses attention over the current sequence (self-attention) to extract 'memories'.
    Usually implies a larger context window or specific attention patterns,
    but here we use a standard TransformerEncoder as the memory block module.
    """
    def __init__(self, d_model, memory_size=128):
        # memory_size not directly used for transformer width unless we project
        super().__init__(d_model)
        # Keep internal dimension d_model for simplicity
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        
    def forward(self, x, state=None):
        # x: (b, seq, d)
        # Transformer is stateless between calls in this simple version
        out = self.transformer(x)
        return out, state

class TITANS(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, output_dim=1, memory_type='neural', memory_size=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Short-term Memory (Core Transformer)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.core_transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Long-term Memory Choice
        self.memory_type = memory_type
        if memory_type == 'neural':
            self.memory = NeuralMemory(d_model, memory_size)
        elif memory_type == 'lstm':
            self.memory = LSTMMemory(d_model, memory_size)
        elif memory_type == 'transformer':
            self.memory = TransformerMemory(d_model, memory_size)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
            
        self.gate = nn.Linear(d_model * 2, d_model)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, seq, dim)
        x_emb = self.input_proj(x)
        
        # STM Branch
        stm_out = self.core_transformer(x_emb)
        
        # LTM Branch
        ltm_out, _ = self.memory(x_emb)
        
        # Gating / Combination
        # We can concat, add, or gate. Titans uses gating.
        combined = torch.cat([stm_out, ltm_out], dim=-1)
        gate_val = torch.sigmoid(self.gate(combined))
        
        # Fused representation
        fused = gate_val * stm_out + (1 - gate_val) * ltm_out
        
        # Prediction
        last_step = fused[:, -1, :]
        return self.decoder(last_step)

def create_model(settings):
    """
    Factory function to create a model based on settings.
    """
    model_type = settings["model"]["type"]
    input_dim = settings["model"]["input_dim"]
    output_dim = settings["model"]["output_dim"]
    
    if model_type == "Linear":
        return LinearRegressionModel(input_dim, output_dim)
    
    elif model_type == "Exponential":
        return ExponentialRegressionModel(input_dim, output_dim)
        
    elif model_type == "FFT":
        seq_len = settings["data"]["sequence_length"]
        # FFT specific configs could be added to settings if needed
        return FFTModel(seq_len=seq_len, input_dim=input_dim, pred_len=output_dim, use_all_features=True)
        
    elif model_type == "Transformer":
        return VanillaTransformer(
            input_dim=input_dim,
            d_model=settings["model"]["d_model"],
            nhead=settings["model"]["nhead"],
            num_layers=settings["model"]["num_layers"],
            output_dim=output_dim,
            dropout=settings["model"]["dropout"]
        )
        
    elif model_type == "TITANS":
        return TITANS(
            input_dim=input_dim,
            d_model=settings["model"]["d_model"],
            nhead=settings["model"]["nhead"],
            num_layers=settings["model"]["num_layers"],
            output_dim=output_dim,
            memory_type=settings["model"]["memory_type"],
            memory_size=settings["model"]["memory_size"]
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
