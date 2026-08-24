import torch
import torch.nn as nn


class QPResolutionInjectionBlock(nn.Module):
    """
    Injection Block for QP (Quantization Parameter) and Resolution metadata.

    Transforms QP scalar/tensor and Image Resolution (Height, Width) into an
    embedding vector via Linear FC layers + PReLU activation, then spatially
    broadcasts and concatenates the embedding with input feature maps.

    Paper Reference:
        "NN-VVC: Versatile Video Coding boosted by self-supervisedly learned
         image coding for machines", Section IV-B (Intra Human Adapter).
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim

        # Process QP: [B, 1] -> [B, embed_dim // 2]
        self.qp_fc = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.PReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.PReLU(),
        )

        # Process Resolution [H, W]: [B, 2] -> [B, embed_dim // 2]
        self.res_fc = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.PReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.PReLU(),
        )

    def forward(
        self,
        qp: torch.Tensor,
        resolution: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            qp: Tensor of shape [B, 1] or [B] containing QP values (e.g. 22..47).
            resolution: Tensor of shape [B, 2] containing [height, width] normalized/raw.

        Returns:
            injection_vector: Tensor of shape [B, embed_dim].
        """
        if qp.dim() == 1:
            qp = qp.unsqueeze(-1)
        if resolution.dim() == 1:
            resolution = resolution.unsqueeze(0)

        # Normalize QP and Resolution to reasonable scales for numerical stability
        qp_norm = qp.float() / 63.0
        res_norm = resolution.float() / 4096.0

        qp_embed = self.qp_fc(qp_norm)
        res_embed = self.res_fc(res_norm)

        injection_vector = torch.cat([qp_embed, res_embed], dim=1)
        return injection_vector

    def broadcast_and_concat(
        self,
        x: torch.Tensor,
        injection_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Spatially broadcast injection vector across feature map height & width,
        then concatenate along channel dimension.

        Args:
            x: Feature map tensor of shape [B, C, H, W].
            injection_vector: Tensor of shape [B, embed_dim].

        Returns:
            Concatenated feature map tensor of shape [B, C + embed_dim, H, W].
        """
        B, C, H, W = x.shape
        inj_spatial = injection_vector.unsqueeze(-1).unsqueeze(-1).expand(B, self.embed_dim, H, W)
        return torch.cat([x, inj_spatial], dim=1)
