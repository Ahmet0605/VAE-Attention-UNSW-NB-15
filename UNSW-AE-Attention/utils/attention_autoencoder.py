import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# 🧠 Feature-level Attention Katmanı
# ==========================================================
class AttentionLayer(nn.Module):
    """Her özelliğin önemini öğrenen attention modülü."""
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_weights = self.fc(x)
        attn_weights = attn_weights / (torch.mean(attn_weights, dim=1, keepdim=True) + 1e-8)
        return attn_weights


# ==========================================================
# 🧩 Katman Bazlı Attention Modülü (Layer Attention)
# ==========================================================
class LayerAttention(nn.Module):
    """Encoder katmanlarının önemini öğrenen attention modülü."""
    def __init__(self, num_layers=3):
        super(LayerAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_layers, num_layers),
            nn.Softmax(dim=-1)
        )

    def forward(self, layer_losses):
        """
        layer_losses: her encoder katmanının reconstruction kaybı
        """
        weights = self.fc(layer_losses)
        return weights


# ==========================================================
# 🧩 Encoder
# ==========================================================
class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Linear(input_size, 96)
        self.encoder_2 = nn.Linear(96, 64)
        self.encoder_3 = nn.Linear(64, 32)
        self.encoder_mean = nn.Linear(32, 8)
        self.encoder_var = nn.Linear(32, 8)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        h1 = self.act(self.encoder_1(x))
        h2 = self.act(self.encoder_2(h1))
        h3 = self.act(self.encoder_3(h2))
        mean = self.encoder_mean(h3)
        log_var = self.encoder_var(h3)
        return [h1, h2, h3], mean, log_var


# ==========================================================
# 🔄 Decoder
# ==========================================================
class Decoder(nn.Module):
    def __init__(self, output_size):
        super(Decoder, self).__init__()
        self.decoder_1 = nn.Linear(8, 32)
        self.decoder_2 = nn.Linear(32, 64)
        self.decoder_3 = nn.Linear(64, 96)
        self.decoder_4 = nn.Linear(96, output_size)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, z):
        z = self.act(self.decoder_1(z))
        z = self.act(self.decoder_2(z))
        z = self.act(self.decoder_3(z))
        out = torch.sigmoid(self.decoder_4(z))
        return out


# ==========================================================
# 🧠 Ana Model: Layer-Aware Attention-VAE
# ==========================================================
class AttentionVAE(nn.Module):
    def __init__(self, input_size):
        super(AttentionVAE, self).__init__()
        self.encoder = Encoder(input_size)
        self.decoder = Decoder(input_size)
        self.feature_attention = AttentionLayer(input_size)
        self.layer_attention = LayerAttention(num_layers=3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        # --- Feature-level attention ---
        attn_weights = self.feature_attention(x)
        x_attn = x * attn_weights

        # --- Encoder ---
        layer_outputs, mean, log_var = self.encoder(x_attn)
        z = self.reparameterize(mean, log_var)

        # --- Decoder ---
        recon = self.decoder(z)

        # --- Katman bazlı attention loss hesaplama ---
        # Her encoder çıkışına reconstruction uygula
        layer_losses = []
        for layer_out in layer_outputs:
            layer_recon = self.decoder(self.reparameterize(mean, log_var))
            loss = torch.mean((layer_out - layer_recon[:, :layer_out.shape[1]]) ** 2, dim=1)
            layer_losses.append(loss.unsqueeze(1))

        # (batch, num_layers)
        layer_losses = torch.cat(layer_losses, dim=1)
        layer_attn_weights = self.layer_attention(layer_losses)

        # --- Katman losslarını attention ile ağırlıkla ---
        weighted_layer_loss = torch.sum(layer_losses * layer_attn_weights, dim=1).mean()

        return recon, mean, log_var, attn_weights, weighted_layer_loss

