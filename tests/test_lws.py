import unittest
from src.training.lws import LWSScheduler, psi


class TestLWSScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = LWSScheduler()

    def test_psi_function(self):
        # psi(0, y) should be 0
        self.assertEqual(psi(0, 1.01), 0.0)
        # psi(x <= 0) should be 0
        self.assertEqual(psi(-5, 1.01), 0.0)
        # psi(1, 1.01) = 1e-3 * (1.01^1 - 1) = 1e-3 * 0.01 = 1e-5
        self.assertAlmostEqual(psi(1, 1.01), 1e-5, places=8)

    def test_w_mse(self):
        for epoch in [0, 50, 100, 320]:
            self.assertEqual(self.scheduler.get_w_mse(epoch), 1.0)

    def test_w_task_schedule(self):
        # Epoch < 50: w_task should be 0
        self.assertEqual(self.scheduler.get_w_task(0), 0.0)
        self.assertEqual(self.scheduler.get_w_task(49), 0.0)

        # Epoch >= 50: w_task should be 4 * psi(epoch - 50, 1.01)
        # At epoch 50: 4 * psi(0, 1.01) = 0
        self.assertEqual(self.scheduler.get_w_task(50), 0.0)
        # At epoch 68 (n=68, offset=18): 4 * psi(18, 1.01)
        expected_w_task_68 = 4.0 * 1e-3 * (pow(1.01, 18) - 1.0)
        self.assertAlmostEqual(self.scheduler.get_w_task(68), expected_w_task_68, places=8)

    def test_w_rate_schedule(self):
        # Epoch < 50: 0.01
        self.assertEqual(self.scheduler.get_w_rate(0), 0.01)
        self.assertEqual(self.scheduler.get_w_rate(49), 0.01)

        # 50 <= Epoch < 62: 0.0
        self.assertEqual(self.scheduler.get_w_rate(50), 0.0)
        self.assertEqual(self.scheduler.get_w_rate(61), 0.0)

        # 62 <= Epoch < 85: 2 * psi(epoch - 62, 1.01)
        self.assertEqual(self.scheduler.get_w_rate(62), 0.0)
        expected_w_rate_68 = 2.0 * 1e-3 * (pow(1.01, 68 - 62) - 1.0)
        self.assertAlmostEqual(self.scheduler.get_w_rate(68), expected_w_rate_68, places=8)

        # 85 <= Epoch < 107: Constant C = 2 * psi(22, 1.01)
        expected_c = 2.0 * 1e-3 * (pow(1.01, 22) - 1.0)
        self.assertAlmostEqual(self.scheduler.C, expected_c, places=8)
        self.assertAlmostEqual(self.scheduler.get_w_rate(85), expected_c, places=8)
        self.assertAlmostEqual(self.scheduler.get_w_rate(106), expected_c, places=8)

        # Epoch >= 107: C + 2 * psi(epoch - 107, 1.02)
        expected_w_rate_107 = expected_c  # psi(0, 1.02) = 0
        self.assertAlmostEqual(self.scheduler.get_w_rate(107), expected_w_rate_107, places=8)
        expected_w_rate_170 = expected_c + 2.0 * 1e-3 * (pow(1.02, 170 - 107) - 1.0)
        self.assertAlmostEqual(self.scheduler.get_w_rate(170), expected_w_rate_170, places=8)

    def test_target_qp_checkpoints(self):
        target_map = {
            68: 22,
            80: 27,
            170: 32,
            220: 37,
            270: 42,
            320: 47,
        }
        for epoch, qp in target_map.items():
            self.assertTrue(self.scheduler.is_checkpoint_epoch(epoch))
            self.assertEqual(self.scheduler.get_target_qp(epoch), qp)
            info = self.scheduler.get_checkpoint_info(epoch)
            self.assertIsNotNone(info)
            self.assertEqual(info["target_qp"], qp)

        # Non-checkpoint epoch
        self.assertFalse(self.scheduler.is_checkpoint_epoch(50))
        self.assertIsNone(self.scheduler.get_target_qp(50))
        self.assertIsNone(self.scheduler.get_checkpoint_info(50))


if __name__ == "__main__":
    unittest.main()
