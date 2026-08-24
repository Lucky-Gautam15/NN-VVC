import torch
import torch.nn as nn

from src.adapters.injection import QPResolutionInjectionBlock


class IntraHumanAdapter(nn.Module):
    """
    Intra Human Adapter (IHA) module from NN-VVC Section IV-B.

    Adapts the reconstructed LIC image representation for human visual perception.

    Key Architectural Features:
        1. Convolutional Autoencoder with encoder & decoder paths.
        2. Local skip connections between matching encoder & decoder stages.
        3. Global input-to-output skip connection (residual learning):
               x_human = clamp(x_input + residual, 0.0, 1.0)
        4. QP & Resolution Injection: QP and spatial resolution are transformed via
           injection blocks, spatially broadcasted, and concatenated before
           down/up convolution operations.
        5. PReLU activation functions throughout.
        6. Training objective: pure MSE loss (w_proxy = 0, w_mse = 1.0).

    Paper Reference:
        "NN-VVC: Versatile Video Coding boosted by self-supervisedly learned
         image coding for machines", Section IV-B.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Metadata injection block for QP and Resolution
        self.injection_block = QPResolutionInjectionBlock(embed_dim=embed_dim)

        # Encoder Path:
        # Stage 1: (3 + embed_dim) -> 64 (stride 2)
        self.enc_conv1 = nn.Conv2d(
            in_channels + embed_dim, 64, kernel_size=5, stride=2, padding=2
        )
        self.enc_act1 = nn.PReLU()

        # Stage 2: (64 + embed_dim) -> 128 (stride 2)
        self.enc_conv2 = nn.Conv2d(
            64 + embed_dim, 128, kernel_size=5, stride=2, padding=2
        )
        self.enc_act2 = nn.PReLU()

        # Bottleneck Residual Block: (128 + embed_dim) -> 128
        self.bottleneck_conv = nn.Conv2d(
            128 + embed_dim, 128, kernel_size=3, padding=1
        )
        self.bottleneck_act = nn.PReLU()

        # Decoder Path (with local skip connections from encoder):
        # Stage 1: (128 [bottleneck] + 128 [enc2_skip] + embed_dim) -> 64 (transpose stride 2)
        self.dec_conv1 = nn.ConvTranspose2d(
            128 + 128 + embed_dim, 64, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.dec_act1 = nn.PReLU()

        # Stage 2: (64 [dec1] + 64 [enc1_skip] + embed_dim) -> 32 (transpose stride 2)
        self.dec_conv2 = nn.ConvTranspose2d(
            64 + 64 + embed_dim, 32, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.dec_act2 = nn.PReLU()

        # Output Residual Layer: 32 -> 3 (3x3 conv)
        self.out_conv = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        qp: torch.Tensor,
        resolution: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input reconstructed image tensor of shape [B, 3, H, W], expected in [0, 1].
            qp: Quantization parameter tensor of shape [B, 1], [B], or scalar float/int.
            resolution: Spatial resolution tensor of shape [B, 2] [height, width].
                        If None, extracted directly from x.shape[-2:].

        Returns:
            x_human: Human-adapted reconstructed image tensor of shape [B, 3, H, W] in [0, 1].
        """
        B, C, H, W = x.shape

        if not isinstance(qp, torch.Tensor):
            qp = torch.tensor([[float(qp)]] * B, device=x.device, dtype=torch.float32)
        elif qp.dim() == 0:
            qp = qp.unsqueeze(0).unsqueeze(0).expand(B, 1)
        elif qp.dim() == 1:
            qp = qp.unsqueeze(1)

        if resolution is None:
            resolution = torch.tensor([[float(H), float(W)]] * B, device=x.device, dtype=torch.float32)
        elif not isinstance(resolution, torch.Tensor):
            resolution = torch.tensor([list(resolution)] * B, device=x.device, dtype=torch.float32)
        elif resolution.dim() == 1:
            resolution = resolution.unsqueeze(0)

        # 1. Compute injection embedding vector: [B, embed_dim]
        inj_vec = self.injection_block(qp=qp, resolution=resolution)

        # 2. Encoder Stage 1
        e1_input = self.injection_block.broadcast_and_concat(x, inj_vec)
        e1 = self.enc_act1(self.enc_conv1(e1_input))  # [B, 64, H/2, W/2]

        # 3. Encoder Stage 2
        e2_input = self.injection_block.broadcast_and_concat(e1, inj_vec)
        e2 = self.enc_act2(self.enc_conv2(e2_input))  # [B, 128, H/4, W/4]

        # 4. Bottleneck Stage
        b_input = self.injection_block.broadcast_and_concat(e2, inj_vec)
        b = self.bottleneck_act(self.bottleneck_conv(b_input))  # [B, 128, H/4, W/4]

        # 5. Decoder Stage 1 (concatenates bottleneck + encoder 2 skip feature)
        d1_input_features = torch.cat([b, e2], dim=1)  # [B, 256, H/4, W/4]
        d1_input = self.injection_block.broadcast_and_concat(d1_input_features, inj_vec)
        d1 = self.dec_act1(self.dec_conv1(d1_input))  # [B, 64, H/2, W/2]

        # 6. Decoder Stage 2 (concatenates decoder 1 + encoder 1 skip feature)
        d2_input_features = torch.cat([d1, e1], dim=1)  # [B, 128, H/2, W/2]
        d2_input = self.injection_block.broadcast_and_concat(d2_input_features, inj_vec)
        d2 = self.dec_act2(self.dec_conv2(d2_input))  # [B, 32, H, W]

        # 7. Output residual map
        residual = self.out_conv(d2)  # [B, 3, H, W]

        # 8. Global input-to-output skip connection
        x_human = torch.clamp(x + residual, min=0.0, max=1.0)

        return x_human
