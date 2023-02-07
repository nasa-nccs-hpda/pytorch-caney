import torch
import pytorch_lightning
from pytorch_lightning.utilities.cli import LightningCLI


class TerraGPULightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):

        # Trainer - performance
        parser.set_defaults({"trainer.accelerator": "auto"})
        parser.set_defaults({"trainer.devices": "auto"})
        parser.set_defaults({"trainer.auto_select_gpus": True})
        parser.set_defaults({"trainer.precision": 32})

        # Trainer - training
        parser.set_defaults({"trainer.max_epochs": 500})
        parser.set_defaults({"trainer.min_epochs": 1})
        parser.set_defaults({"trainer.detect_anomaly": True})
        parser.set_defaults({"trainer.logger": True})
        parser.set_defaults({"trainer.default_root_dir": "output_model"})

        # Trainer - optimizer - TODO
        default_optimizer = {
            "class_path": torch.optim.Adam,
            "init_args": {
                "lr": 0.01
            }
        }

        # Trainer - callbacks
        default_callbacks = [
            {"class_path": "pytorch_lightning.callbacks.DeviceStatsMonitor"},
            {
                "class_path": "pytorch_lightning.callbacks.EarlyStopping",
                "init_args": {
                    "monitor": "val_loss",
                    "patience": 5,
                    "mode": "min"
                }
            },
            #{
            #    "class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
            #    "init_args": {
            #        "dirpath": "output_model",
            #        "monitor": "val_loss",
            #        "auto_insert_metric_name": True
            #    }
            #},
        ]
        parser.set_defaults({"trainer.callbacks": default_callbacks})


        #    {
        #        "class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
        #        "init_args": {
        #            "dirpath": "output_model",
        #            "monitor": "val_loss",
        #            "auto_insert_metric_name": True
        #        }
        #    },
        # ]
        # parser.set_defaults({"trainer.callbacks": default_callbacks})
