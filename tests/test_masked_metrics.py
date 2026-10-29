import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_caney.configs.config import _C
from pytorch_caney.pipelines.three_d_cloud_track_pipeline import ThreeDCloudTrackTask
from pytorch_caney.three_d_cloud_track_utils import (
    IGNORE_INDEX,
    build_masked_metric_tensors,
    masked_binary_cross_entropy_with_logits,
)


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, x):
        batch = x.shape[0]
        return self.logits.expand(batch, -1, -1, -1)


class SingleBatchDataset(Dataset):
    def __init__(self, inputs, labels, valid):
        self.inputs = inputs
        self.labels = labels
        self.valid = valid

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index], self.valid[index]


def make_pipeline_config():
    config = _C.clone()
    config.defrost()
    config.DATA.IMG_SIZE = 2
    config.MODEL.IN_CHANS = 1
    config.MODEL.ENCODER = 'fcn'
    config.MODEL.DECODER = 'denseprofiledecoder'
    config.MODEL.HEAD = 'dense_profile_head'
    config.MODEL.OUTPUT_BINS = 1
    config.LOSS.NAME = 'bce'
    config.freeze()
    return config


def test_masked_loss_ignores_invalid_positions_and_zeroes_gradients():
    logits = torch.tensor([[[[0.0, 2.0], [4.0, -3.0]]]], requires_grad=True)
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    valid = torch.tensor([[[[True, False], [True, False]]]])

    loss, _ = masked_binary_cross_entropy_with_logits(logits, targets, valid)
    loss.backward()

    alt_logits = logits.detach().clone()
    alt_logits[..., 0, 1] = -99.0
    alt_logits[..., 1, 1] = 99.0
    alt_loss, _ = masked_binary_cross_entropy_with_logits(alt_logits, targets, valid)

    assert torch.allclose(loss.detach(), alt_loss.detach())
    assert logits.grad[..., 0, 1].item() == 0.0
    assert logits.grad[..., 1, 1].item() == 0.0
    assert logits.grad[..., 0, 0].abs().item() > 0.0
    assert logits.grad[..., 1, 0].abs().item() > 0.0


def test_iou_metric_ignores_invalid_bins():
    logits = torch.tensor([0.9, 0.1, 0.7, 0.2])
    targets = torch.tensor([1, 0, 1, 0])
    valid = torch.tensor([True, True, False, False])
    preds, metric_targets = build_masked_metric_tensors(
        logits.view(1, 1, 2, 2),
        targets.view(1, 1, 2, 2),
        valid.view(1, 1, 2, 2),
    )

    assert metric_targets.tolist() == [1, 0, IGNORE_INDEX, IGNORE_INDEX]


def test_validation_epoch_metrics_ignore_invalid_bins():
    config = make_pipeline_config()
    model = ThreeDCloudTrackTask(config)
    fixed_logits = torch.tensor([[[[8.0, -8.0], [8.0, -8.0]]]])
    model.model = FixedLogitModel(fixed_logits)

    inputs = torch.zeros((1, 1, 2, 2))
    labels = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    valid = torch.tensor([[[[True, True], [False, False]]]])
    loader = DataLoader(SingleBatchDataset(inputs, labels, valid), batch_size=1)

    trainer = type('TrainerStub', (), {'num_sanity_val_steps': 0})()
    model.trainer = trainer
    model.on_validation_epoch_start()
    batch = next(iter(loader))
    model.validation_step(batch, 0)

    assert torch.isclose(model.val_iou.compute(), torch.tensor(1.0))
    assert torch.isclose(model.val_precision.compute(), torch.tensor(1.0))
    assert torch.isclose(model.val_recall.compute(), torch.tensor(1.0))
    assert torch.isclose(model.val_f1.compute(), torch.tensor(1.0))
