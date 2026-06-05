import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # V: embeddings para a palavra central (input vectors)
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)

        # U: embeddings para a palavra de contexto (output vectors)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # Opcional: inicialização simples (não é obrigatório)
        nn.init.uniform_(self.in_embed.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(
        self,
        center_ids: torch.Tensor,       # (B,)
        pos_context_ids: torch.Tensor,  # (B,)
        neg_context_ids: torch.Tensor,  # (B, K)
    ) -> torch.Tensor:
        
        """
        Calcula a loss do modelo Word2Vec Skip-Gram com Negative Sampling.
        Args:
            center_ids: Tensor de IDs das palavras centrais, shape (B,)
            pos_context_ids: Tensor de IDs das palavras de contexto positivas, shape (B,)
            neg_context_ids: Tensor de IDs das palavras de contexto negativas, shape (B, K)
        Returns:
            loss: Tensor escalar representando a loss média do batch
        """
        center_ids = center_ids.long()
        pos_context_ids = pos_context_ids.long()
        neg_context_ids = neg_context_ids.long()

        v = self.in_embed(center_ids)           # (B, D)
        u_pos = self.out_embed(pos_context_ids) # (B, D)
        u_neg = self.out_embed(neg_context_ids) # (B, K, D)

        # score positivo: u_pos · v
        pos_score = torch.sum(v * u_pos, dim=1)              # (B,)
        pos_loss = F.logsigmoid(pos_score)                   # (B,)

        # score negativo: u_neg · v  (faz broadcast do v para (B, K, D))
        neg_score = torch.sum(u_neg * v.unsqueeze(1), dim=2) # (B, K)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)       # (B,)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_input_embeddings(self) -> torch.Tensor:
        """Retorna os embeddings de entrada (V)."""
        return self.in_embed.weight

    def get_output_embeddings(self) -> torch.Tensor:
        """Retorna os embeddings de saída (U)."""
        return self.out_embed.weight