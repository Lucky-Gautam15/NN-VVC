import math
import unittest

import torch
import torch.optim as optim

from src.adapters.iha import IntraHumanAdapter
from src.adapters.injection import QPResolutionInjectionBlock
from src.losses.mse_loss import MSELoss


class TestIntraHumanAdapter(unittest.TestCase):
    """
    Phase D Verification Test Suite for Intra Human Adapter (IHA):
    - Model construction & initialization
    - Forward pass & shape retention [B, 3, H, W]
    - Evaluation across target QPs (22, 27, 32, 37, 42, 47)
    - Evaluation across spatial resolutions (128x128, 256x256, 512x512)
    - Verification that QP injection alters network behavior
    - Verification that Resolution injection alters network behavior
    - Finiteness & output range [0, 1] validation
    - Backward gradient flow across all trainable parameters
    - Optimizer step execution
    - Synthetic multi-image MSE loss training regression
    """

    def setUp(self):
        self.model = IntraHumanAdapter(in_channels=3, embed_dim=16)

    def test_d_1_construction_and_forward_shapes(self):
        """D.1 & D.3: Model construction, forward pass, and shape preservation."""
        x = torch.rand(2, 3, 256, 256)
        qp = torch.tensor([22.0, 32.0])
        out = self.model(x, qp)

        self.assertEqual(out.shape, torch.Size([2, 3, 256, 256]))
        self.assertEqual(out.dtype, torch.float32)

    def test_d_4_target_qps_evaluation(self):
        """D.4: Verification across all 6 target VVC QPs (22, 27, 32, 37, 42, 47)."""
        target_qps = [22, 27, 32, 37, 42, 47]
        x = torch.rand(1, 3, 256, 256)

        for qp in target_qps:
            out = self.model(x, qp=qp)
            self.assertEqual(out.shape, x.shape)
            self.assertFalse(torch.isnan(out).any())
            self.assertFalse(torch.isinf(out).any())

    def test_d_5_different_spatial_resolutions(self):
        """D.5: Verification across different spatial resolutions."""
        resolutions = [(128, 128), (256, 256), (384, 512)]

        for H, W in resolutions:
            x = torch.rand(1, 3, H, W)
            out = self.model(x, qp=27)
            self.assertEqual(out.shape, torch.Size([1, 3, H, W]))
            self.assertFalse(torch.isnan(out).any())

    def test_d_6_qp_injection_effect(self):
        """D.6: Verify that changing QP injection vector actually alters the model output."""
        x = torch.ones(1, 3, 128, 128) * 0.5
        res = torch.tensor([[128.0, 128.0]])

        out_qp22 = self.model(x, qp=22, resolution=res)
        out_qp47 = self.model(x, qp=47, resolution=res)

        diff = (out_qp22 - out_qp47).abs().mean().item()
        self.assertGreater(diff, 1e-6, "QP injection had no measurable effect on network output")

    def test_d_7_resolution_injection_effect(self):
        """D.7: Verify that changing Resolution injection vector alters the model output."""
        x = torch.ones(1, 3, 128, 128) * 0.5
        qp = torch.tensor([[32.0]])

        res_small = torch.tensor([[128.0, 128.0]])
        res_large = torch.tensor([[1080.0, 1920.0]])

        out_small = self.model(x, qp=qp, resolution=res_small)
        out_large = self.model(x, qp=qp, resolution=res_large)

        diff = (out_small - out_large).abs().mean().item()
        self.assertGreater(diff, 1e-6, "Resolution injection had no measurable effect on output")

    def test_d_8_d_9_output_range_and_finiteness(self):
        """D.8 & D.9: Bounded output range [0, 1] and non-NaN/non-Inf finiteness."""
        x = torch.randn(2, 3, 128, 128)  # Out of range input to stress test clamp
        out = self.model(x, qp=32)

        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        self.assertGreaterEqual(float(out.detach().min()), 0.0)
        self.assertLessEqual(float(out.detach().max()), 1.0)

    def test_d_10_d_11_backward_gradients_and_optimizer_step(self):
        """D.10 & D.11: Complete backward pass, gradient presence/finiteness, and optimizer update."""
        model = IntraHumanAdapter(in_channels=3, embed_dim=16)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        mse_loss_fn = MSELoss()

        x_rec = torch.rand(2, 3, 128, 128, requires_grad=True)
        x_gt = torch.rand(2, 3, 128, 128)
        qp = torch.tensor([27.0, 37.0])

        x_human = model(x_rec, qp=qp)
        loss = mse_loss_fn(x_human, x_gt)

        optimizer.zero_grad()
        loss.backward()

        param_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count += 1
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN")
                self.assertFalse(torch.isinf(param.grad).any(), f"Parameter {name} gradient contains Inf")

        self.assertGreater(param_count, 0)
        optimizer.step()

    def test_d_12_synthetic_regression_training(self):
        """D.12: Synthetic multi-image regression training loop using pure MSE objective (w_proxy = 0)."""
        model = IntraHumanAdapter(in_channels=3, embed_dim=16)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        mse_loss_fn = MSELoss()

        x_gt = torch.rand(4, 3, 128, 128)
        x_rec = (x_gt + 0.05 * torch.randn_like(x_gt)).clamp(0.0, 1.0)
        qp = torch.tensor([22.0, 27.0, 32.0, 42.0])

        initial_loss = float(mse_loss_fn(model(x_rec, qp), x_gt).item())

        for step in range(5):
            optimizer.zero_grad()
            out = model(x_rec, qp)
            loss = mse_loss_fn(out, x_gt)
            loss.backward()
            optimizer.step()

        final_loss = float(mse_loss_fn(model(x_rec, qp), x_gt).item())
        self.assertLessEqual(final_loss, initial_loss, "IHA synthetic MSE training did not reduce loss")


if __name__ == "__main__":
    unittest.main()
