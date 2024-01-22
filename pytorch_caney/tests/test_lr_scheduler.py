from pytorch_caney.lr_scheduler import (
    build_scheduler,
)

import unittest
from unittest.mock import (
    Mock,
    patch,
)


class TestBuildScheduler(unittest.TestCase):
    def setUp(
        self,
    ):
        self.config = Mock(
            TRAIN=Mock(
                EPOCHS=300,
                WARMUP_EPOCHS=20,
                MIN_LR=1e-6,
                WARMUP_LR=1e-7,
                LR_SCHEDULER=Mock(
                    NAME="cosine",
                    DECAY_EPOCHS=30,
                    DECAY_RATE=0.1,
                    MULTISTEPS=[
                        50,
                        100,
                    ],
                    GAMMA=0.1,
                ),
            )
        )

        self.optimizer = Mock()
        self.n_iter_per_epoch = 100  # Example value

    def test_build_cosine_scheduler(
        self,
    ):
        with patch(
            "pytorch_caney.lr_scheduler.CosineLRScheduler"
        ) as mock_cosine_scheduler:
            _ = build_scheduler(
                self.config,
                self.optimizer,
                self.n_iter_per_epoch,
            )

        mock_cosine_scheduler.assert_called_once_with(
            self.optimizer,
            t_initial=300 * 100,
            cycle_mul=1.0,
            lr_min=1e-6,
            warmup_lr_init=1e-7,
            warmup_t=20 * 100,
            cycle_limit=1,
            t_in_epochs=False,
        )


if __name__ == "__main__":
    unittest.main()
