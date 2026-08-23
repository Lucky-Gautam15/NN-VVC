import math
from typing import Dict, Optional, Tuple


def psi(x: float, y: float) -> float:
    """
    Exponential scaling function defined in NN-VVC paper (Equation 5):
        psi(x, y) = 10^-3 * (y^x - 1)
    """
    if x <= 0:
        return 0.0
    return 1e-3 * (math.pow(y, x) - 1.0)


class LWSScheduler:
    """
    Loss Weighting Strategy (LWS) scheduler from the NN-VVC reference paper:
    "NN-VVC: Versatile Video Coding boosted by self-supervisedly learned
     image coding for machines", Section IV-A, Equation 5.

    Formulation:
        w_mse = 1
        w_task(n) =
            0,                     n < p1
            4 * psi(n - p1, 1.01), n >= p1

        w_rate(n) =
            0.01,                                  n < p1
            0,                                     p1 <= n < p2
            2 * psi(n - p2, 1.01),                 p2 <= n < p3
            c,                                     p3 <= n < p4
            c + 2 * psi(n - p4, 1.02),             n >= p4

    Constants:
        p1 = 50, p2 = 62, p3 = 85, p4 = 107
        c = 2 * psi(p3 - p2 - 1, 1.01) = 2 * psi(22, 1.01)

    Target Checkpoints:
        Epoch 68  -> Target VVC QP 22
        Epoch 80  -> Target VVC QP 27
        Epoch 170 -> Target VVC QP 32
        Epoch 220 -> Target VVC QP 37
        Epoch 270 -> Target VVC QP 42
        Epoch 320 -> Target VVC QP 47
    """

    P1: int = 50
    P2: int = 62
    P3: int = 85
    P4: int = 107

    # Constant c = 2 * psi(p3 - p2 - 1, 1.01) = 2 * psi(22, 1.01)
    C: float = 2.0 * psi(85 - 62 - 1, 1.01)

    # 6 Target Checkpoints specified by paper (epoch -> target QP)
    QP_CHECKPOINTS: Dict[int, int] = {
        68: 22,
        80: 27,
        170: 32,
        220: 37,
        270: 42,
        320: 47,
    }

    def __init__(self):
        pass

    def get_w_mse(self, epoch: int) -> float:
        """w_mse = 1 for all epochs."""
        return 1.0

    def get_w_task(self, epoch: int) -> float:
        """
        w_task(n) =
            0,                     n < 50
            4 * psi(n - 50, 1.01), n >= 50
        """
        if epoch < self.P1:
            return 0.0
        return 4.0 * psi(float(epoch - self.P1), 1.01)

    def get_w_rate(self, epoch: int) -> float:
        """
        w_rate(n) =
            0.01,                                  n < 50
            0,                                     50 <= n < 62
            2 * psi(n - 62, 1.01),                 62 <= n < 85
            c,                                     85 <= n < 107
            c + 2 * psi(n - 107, 1.02),            n >= 107
        """
        if epoch < self.P1:
            return 0.01
        elif epoch < self.P2:
            return 0.0
        elif epoch < self.P3:
            return 2.0 * psi(float(epoch - self.P2), 1.01)
        elif epoch < self.P4:
            return self.C
        else:
            return self.C + 2.0 * psi(float(epoch - self.P4), 1.02)

    def get_weights(self, epoch: int) -> Tuple[float, float, float]:
        """
        Return (w_rate, w_mse, w_task) for a given epoch number n (1-indexed or 0-indexed integer).
        """
        w_rate = self.get_w_rate(epoch)
        w_mse = self.get_w_mse(epoch)
        w_task = self.get_w_task(epoch)
        return w_rate, w_mse, w_task

    def is_checkpoint_epoch(self, epoch: int) -> bool:
        """Check if epoch is one of the 6 target QP checkpoints."""
        return epoch in self.QP_CHECKPOINTS

    def get_target_qp(self, epoch: int) -> Optional[int]:
        """Return target QP if epoch is a checkpoint epoch, else None."""
        return self.QP_CHECKPOINTS.get(epoch, None)

    def get_checkpoint_info(self, epoch: int) -> Optional[Dict]:
        """Return detailed info dictionary for target checkpoint epochs."""
        if not self.is_checkpoint_epoch(epoch):
            return None
        w_rate, w_mse, w_task = self.get_weights(epoch)
        return {
            "epoch": epoch,
            "target_qp": self.QP_CHECKPOINTS[epoch],
            "w_rate": w_rate,
            "w_mse": w_mse,
            "w_task": w_task,
        }
