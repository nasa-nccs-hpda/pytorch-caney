import unittest
import argparse
import rioxarray as rxr
import segmentation_models_pytorch as smp

from pytorch_caney.inference import sliding_window_tiler_multiclass


class TestInference(unittest.TestCase):

    def setUp(self):
        # Initialize any required configuration here
        #config_path = 'pytorch_caney/' + \
        #    'tests/config/test_config.yaml'
        #args = argparse.Namespace(cfg=config_path)
        #self.config = get_config(args)
        cog_url = (
            "https://oin-hotosm.s3.amazonaws.com/"
            "5d7dad0becaf880008a9bc88/0/5d7dad0becaf880008a9bc89.tif"
        )
        self.raster = rxr.open_rasterio(cog_url, masked=True, overview_level=4)
        self.raster = self.raster.transpose("y", "x", "band")

    def test_sliding_window_tiler_multiclass(self):

        print(self.raster.shape)

        model = smp.Unet('resnet34', classes=4)

        sliding_window_tiler_multiclass(
            self.raster,
            model,
            n_classes=4,
            img_size=128,
            pad_style='reflect',
            overlap=0.50,
            constant_value=600,
            batch_size=1024,
            threshold=0.50,
            standardization=None,
            mean=None,
            std=None,
            normalize=1.0,
            rescale=None,
            window='triang',  # 'overlap-tile'
            probability_map=False
        )

if __name__ == '__main__':
    unittest.main()
