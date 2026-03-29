import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TRMConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 1365
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    n_latent_steps: int = 6  # n in latent_recursion
    t_deep_steps: int = 3  # T in deep_recursion
    n_supervision_steps: int = 16  # N_sup


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RoPE(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rope = RoPE(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rope(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = SwiGLU(config)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TRMBaseNetwork(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class TRM(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.input_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.base_network = TRMBaseNetwork(config)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.Q_head = nn.Linear(config.hidden_size, 1)  # Binary classification head

    def get_attention_mask(self, seq_len, device):
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def latent_recursion(self, x_embed, y_embed, z):
        # Combines x_embed, y_embed, and previous z, then applies base network n times
        bsz, seq_len, _ = x_embed.shape
        attention_mask = self.get_attention_mask(seq_len * 3, x_embed.device)

        for _ in range(self.config.n_latent_steps):
            # Concatenate along sequence dimension [x, y, z] -> [B, 3L, D]
            combined = torch.cat([x_embed, y_embed, z], dim=1)
            output = self.base_network(combined, attention_mask=attention_mask)
            # Update z (taking the last segment corresponding to z)
            z = output[:, -seq_len:, :]
        return z

    def deep_recursion(self, x_embed, y_embed, z):
        # Wraps latent_recursion for T iterations, saving memory for T-1
        for i in range(self.config.t_deep_steps):
            if i < self.config.t_deep_steps - 1:
                with torch.no_grad():
                    z = self.latent_recursion(x_embed, y_embed, z)
            else:
                z = self.latent_recursion(x_embed, y_embed, z)
        return z

    @torch.no_grad()
    def inference(self, question_ids, max_steps=None):
        if max_steps is None:
            max_steps = self.config.n_supervision_steps

        x_embed = self.input_embedding(question_ids)
        bsz, seq_len, dim = x_embed.shape

        z = torch.zeros_like(x_embed)
        y_embed = torch.zeros_like(x_embed)

        for step in range(max_steps):
            x_embed = self.input_embedding(question_ids)

            z = self.deep_recursion(x_embed, y_embed, z)

            logits = self.output_head(z)  # [B, L, V]
            q_logits = self.Q_head(z.mean(dim=1)).squeeze(-1)  # [B]

            pred_ids = logits.argmax(dim=-1)
            y_embed = self.input_embedding(pred_ids)

            q_probs = torch.sigmoid(q_logits)
            if (q_probs > 0.5).all():
                break

        return pred_ids

    @torch.no_grad()
    def eval_step(self, question_ids, target_ids):
        # Evaluation Loop without optimizer
        x_embed = self.input_embedding(question_ids)
        bsz, seq_len, dim = x_embed.shape

        # Initialize z and y (e.g., zeros)
        z = torch.zeros_like(x_embed)
        y_embed = torch.zeros_like(x_embed)

        total_loss = 0

        for step in range(self.config.n_supervision_steps):
            x_embed = self.input_embedding(question_ids)

            # 1. Forward pass (Deep Recursion)
            z = self.deep_recursion(x_embed, y_embed, z)

            # Predict answer and Q-value
            logits = self.output_head(z)  # [B, L, V]
            q_logits = self.Q_head(z.mean(dim=1)).squeeze(
                -1
            )  # Aggregate sequence for Q, [B]

            pred_ids = logits.argmax(dim=-1)
            y_embed = self.input_embedding(pred_ids)

            # 2. Compute Loss
            loss_ce = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), target_ids.reshape(-1)
            )

            is_correct = (pred_ids == target_ids).all(dim=-1).float()
            loss_bce = F.binary_cross_entropy_with_logits(q_logits, is_correct)

            step_loss = loss_ce + loss_bce
            total_loss += step_loss.item()

            # Early Stopping Check
            q_probs = torch.sigmoid(q_logits)
            if (q_probs > 0.5).all():
                break

        return total_loss

    def train_step(self, question_ids, target_ids, optimizer):
        # Deep Supervision Training Loop
        x_embed = self.input_embedding(question_ids)
        bsz, seq_len, dim = x_embed.shape

        # Initialize z and y (e.g., zeros)
        z = torch.zeros_like(x_embed)
        y_embed = torch.zeros_like(x_embed)

        total_loss = 0

        for step in range(self.config.n_supervision_steps):
            # Recompute x_embed so it has a fresh computation graph for this step
            x_embed = self.input_embedding(question_ids)

            # 1. Forward pass (Deep Recursion)
            z = self.deep_recursion(x_embed, y_embed, z)

            # Predict answer and Q-value
            logits = self.output_head(z)  # [B, L, V]
            q_logits = self.Q_head(z.mean(dim=1)).squeeze(
                -1
            )  # Aggregate sequence for Q, [B]

            # Formulate y_pred for next step (using soft embeddings or hard argmax-embedded)
            # Here we just use the embeddings of the argmax prediction for simplicity
            pred_ids = logits.argmax(dim=-1)
            y_embed = self.input_embedding(pred_ids)

            # 2. Compute Loss
            # CrossEntropy for prediction
            loss_ce = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), target_ids.reshape(-1)
            )

            # Determine if prediction is exactly correct (for Q_head target)
            is_correct = (pred_ids == target_ids).all(dim=-1).float()

            # Binary CrossEntropy for Q_head
            loss_bce = F.binary_cross_entropy_with_logits(q_logits, is_correct)

            step_loss = loss_ce + loss_bce

            # 3. Backpropagation for this step
            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()

            total_loss += step_loss.item()

            # Early Stopping Check
            q_probs = torch.sigmoid(q_logits)
            if (q_probs > 0.5).all():
                break  # All answers predicted as correct

            # 4. Gradient Detachment to prevent full unrolling
            z = z.detach()
            y_embed = y_embed.detach()

        return total_loss


if __name__ == "__main__":
    # Test initialization and a mock training step
    config = TRMConfig()
    model = TRM(config)

    print(
        f"TRM Baseline Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
    )

    # Mock data
    bsz = 2
    seq_len = 32
    q_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    t_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model.train_step(q_ids, t_ids, optimizer)
    print(f"Training step completed with loss: {loss:.4f}")
