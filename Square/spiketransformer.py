class LIFNeuron(nn.Module):
    def __init__(self, threshold, decay):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.voltage = 0

    def forward(self, x):
        spike = th.zeros_like(x)
        self.voltage = self.voltage * self.decay * (1. - spike) + x
        spike[self.voltage > self.threshold] = 1.
        self.voltage[self.voltage > self.threshold] = 0

        return spike

class SpikeSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.neuron = LIFNeuron(threshold=1.0, decay=0.9)

    def forward(self, x):
        q = self.neuron(self.query(x))
        k = self.neuron(self.key(x))
        v = self.neuron(self.value(x))

        attn_output_weights = F.softmax(
            q @ k.transpose(-2, -1) / math.sqrt(self.d_model), dim=-1)
        attn_output = attn_output_weights @ v
        attn_output = self.neuron(attn_output)

        return attn_output

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = th.zeros(max_len, emb_size)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, emb_size, 2).float() * - (math.log(10000.0) / emb_size))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SpikingTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward, dropout)
        self.self_attn = SpikeSelfAttention(d_model, nhead, dropout)

class SpikingViT(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size, num_layers, num_heads, num_classes):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        self.pos_embedding = PositionalEncoding(emb_size, max_len=(img_size//patch_size)**2)
        self.transformer_encoder = nn.TransformerEncoder(SpikingTransformerEncoderLayer(emb_size, num_heads), num_layers)
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x