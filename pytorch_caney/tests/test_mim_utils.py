import unittest
from unittest.mock import Mock, patch
import torch
import numpy as np
from pytorch_caney.training.mim_utils import (build_optimizer,
                                              set_weight_decay,
                                              check_keywords_in_name,
                                              get_pretrain_param_groups,
                                              get_swin_layer,
                                              remap_pretrained_keys_swin,
                                              remap_pretrained_keys_vit,
                                              load_pretrained,
                                              reduce_tensor)



class TestBuildOptimizer(unittest.TestCase):

    def setUp(self):
        self.config = Mock()
        self.config.TRAIN.LAYER_DECAY = 0.8
        self.config.TRAIN.BASE_LR = 0.001
        self.config.TRAIN.WEIGHT_DECAY = 0.05
        self.config.TRAIN.OPTIMIZER.EPS = 1e-8
        self.config.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
        self.model = Mock()
        self.logger = Mock()

    @patch('pytorch_caney.training.mim_utils.get_pretrain_param_groups')
    def test_build_optimizer_pretrain(self, mock_get_pretrain):
        mock_get_pretrain.return_value = [{'params': [torch.nn.Parameter(torch.randn(2, 2))]}]
        
        optimizer = build_optimizer(self.config, self.model, is_pretrain=True, logger=self.logger)
        
        self.assertIsNotNone(optimizer)
        mock_get_pretrain.assert_called_once()
        self.logger.info.assert_called()

    @patch('pytorch_caney.training.mim_utils.get_finetune_param_groups')
    def test_build_optimizer_finetune_swin(self, mock_get_finetune):
        self.config.MODEL.TYPE = 'swin'
        self.config.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
        mock_get_finetune.return_value = [{'params': [torch.nn.Parameter(torch.randn(2, 2))]}]
        
        optimizer = build_optimizer(self.config, self.model, is_pretrain=False, logger=self.logger)
        
        self.assertIsNotNone(optimizer)
        mock_get_finetune.assert_called_once()
        self.logger.info.assert_called()

    @patch('pytorch_caney.training.mim_utils.get_finetune_param_groups')
    def test_build_optimizer_finetune_swinv2(self, mock_get_finetune):
        self.config.MODEL.TYPE = 'swinv2'
        self.config.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
        mock_get_finetune.return_value = [{'params': [torch.nn.Parameter(torch.randn(2, 2))]}]
        
        optimizer = build_optimizer(self.config, self.model, is_pretrain=False, logger=self.logger)
        
        self.assertIsNotNone(optimizer)
        mock_get_finetune.assert_called_once()
        self.logger.info.assert_called()

    @patch('pytorch_caney.training.mim_utils.get_pretrain_param_groups')
    def test_no_weight_decay(self, mock_get_pretrain):
        self.model.no_weight_decay.return_value = {'skip_param'}
        self.model.no_weight_decay_keywords.return_value = {'skip_keyword'}
        mock_get_pretrain.return_value = [{'params': [torch.nn.Parameter(torch.randn(2, 2))]}]
        
        build_optimizer(self.config, self.model, is_pretrain=True, logger=self.logger)
        
        self.model.no_weight_decay.assert_called_once()
        self.model.no_weight_decay_keywords.assert_called_once()
        mock_get_pretrain.assert_called_once()


class TestSetWeightDecay(unittest.TestCase):

    def setUp(self):
        self.model = Mock()

    def test_basic_weight_decay(self):
        # Create a mock model with some parameters
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(1))
        param3 = torch.nn.Parameter(torch.randn(5, 5))
        
        self.model.named_parameters.return_value = [
            ('layer1.weight', param1),
            ('layer1.bias', param2),
            ('layer2.weight', param3)
        ]

        result = set_weight_decay(self.model)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]['params']), 2)  # has_decay
        self.assertEqual(len(result[1]['params']), 1)  # no_decay
        self.assertEqual(result[1]['weight_decay'], 0.)

    def test_skip_list(self):
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(5, 5))
        
        self.model.named_parameters.return_value = [
            ('layer1.weight', param1),
            ('layer2.weight', param2)
        ]

        result = set_weight_decay(self.model, skip_list=('layer1.weight',))

        self.assertEqual(len(result[0]['params']), 1)  # has_decay
        self.assertEqual(len(result[1]['params']), 1)  # no_decay

    def test_skip_keywords(self):
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(5, 5))
        
        self.model.named_parameters.return_value = [
            ('layer1_skip.weight', param1),
            ('layer2.weight', param2)
        ]

        result = set_weight_decay(self.model, skip_keywords=('skip',))

        self.assertEqual(len(result[0]['params']), 1)  # has_decay
        self.assertEqual(len(result[1]['params']), 1)  # no_decay

    def test_frozen_weights(self):
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(5, 5))
        param2.requires_grad = False
        
        self.model.named_parameters.return_value = [
            ('layer1.weight', param1),
            ('layer2.weight', param2)
        ]

        result = set_weight_decay(self.model)

        self.assertEqual(len(result[0]['params']), 1)  # has_decay
        self.assertEqual(len(result[1]['params']), 0)  # no_decay


class TestCheckKeywordsInName(unittest.TestCase):

    def test_no_keywords(self):
        self.assertFalse(check_keywords_in_name("test_name"))

    def test_keyword_match(self):
        self.assertTrue(check_keywords_in_name("test_name", ("test",)))

    def test_keyword_no_match(self):
        self.assertFalse(check_keywords_in_name("test_name", ("keyword",)))

    def test_multiple_keywords(self):
        self.assertTrue(check_keywords_in_name("test_name", ("test", "other")))
        self.assertFalse(check_keywords_in_name("test_name", ("keyword1", "keyword2")))

class TestGetPretrainParamGroups(unittest.TestCase):

    def setUp(self):
        self.model = Mock()

    def test_param_grouping(self):
        # Create mock parameters
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(1))
        param3 = torch.nn.Parameter(torch.randn(5, 5))
        param4 = torch.nn.Parameter(torch.randn(3, 3))
        param4.requires_grad = False

        # Set up mock model's named_parameters
        self.model.named_parameters.return_value = [
            ('layer1.weight', param1),
            ('layer1.bias', param2),
            ('layer2.weight', param3),
            ('layer3.weight', param4)
        ]

        result = get_pretrain_param_groups(self.model)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]['params']), 2)  # has_decay
        self.assertEqual(len(result[1]['params']), 1)  # no_decay
        self.assertEqual(result[1]['weight_decay'], 0.)

    def test_skip_list(self):
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(5, 5))

        self.model.named_parameters.return_value = [
            ('layer1.weight', param1),
            ('layer2.weight', param2)
        ]

        result = get_pretrain_param_groups(self.model, skip_list=('layer1.weight',))

        self.assertEqual(len(result[0]['params']), 1)  # has_decay
        self.assertEqual(len(result[1]['params']), 1)  # no_decay

    def test_skip_keywords(self):
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(5, 5))

        self.model.named_parameters.return_value = [
            ('layer1_skip.weight', param1),
            ('layer2.weight', param2)
        ]

        result = get_pretrain_param_groups(self.model, skip_keywords=('skip',))

        self.assertEqual(len(result[0]['params']), 1)  # has_decay
        self.assertEqual(len(result[1]['params']), 1)  # no_decay



class TestGetSwinLayer(unittest.TestCase):
    """
    Test suite for the get_swin_layer function.

    This test suite covers various scenarios for the get_swin_layer function:
    1. Testing special cases like 'mask_token' and 'patch_embed'
    2. Testing layer identification for different parts of the Swin Transformer
    3. Testing the default case for unknown layer names

    The tests use a sample depths configuration of [2, 2, 6, 2] and a total of 14 layers
    (sum of depths + 2) to simulate a realistic Swin Transformer architecture.
    """

    def setUp(self):
        self.num_layers = 14  # sum of depths (2+2+6+2) + 2
        self.depths = [2, 2, 6, 2]

    def test_special_cases(self):
        self.assertEqual(get_swin_layer("mask_token", self.num_layers, self.depths), 0)
        self.assertEqual(get_swin_layer("patch_embed", self.num_layers, self.depths), 0)

    def test_layers(self):
        self.assertEqual(get_swin_layer("layers.0.0.norm", self.num_layers, self.depths), 2)
        self.assertEqual(get_swin_layer("layers.1.1.norm", self.num_layers, self.depths), 4)
        self.assertEqual(get_swin_layer("layers.2.5.norm", self.num_layers, self.depths), 10)
        self.assertEqual(get_swin_layer("layers.3.1.reduction", self.num_layers, self.depths), 12)

    def test_reduction_and_norm(self):
        self.assertEqual(get_swin_layer("layers.0.1.reduction", self.num_layers, self.depths), 2)
        self.assertEqual(get_swin_layer("layers.1.1.norm", self.num_layers, self.depths), 4)
        self.assertEqual(get_swin_layer("layers.2.2.norm", self.num_layers, self.depths), 10)
        self.assertEqual(get_swin_layer("layers.3.2.norm", self.num_layers, self.depths), 12)

    def test_unknown_layer(self):
        self.assertEqual(get_swin_layer("unknown_layer", self.num_layers, self.depths), 13)


class TestReduceTensor(unittest.TestCase):
    """
    Test suite for the reduce_tensor function.

    This test covers:
    1. Tensor reduction across distributed processes
    2. Correct averaging of the reduced tensor
    """

    @patch('torch.distributed.all_reduce')
    @patch('torch.distributed.get_world_size')
    def test_reduce_tensor(self, mock_get_world_size, mock_all_reduce):
        # Setup
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        mock_get_world_size.return_value = 4

        # Mock the all_reduce function to simulate summing across processes
        def mock_all_reduce_func(tensor, op):
            tensor *= 4  # Simulate summing across 4 processes

        mock_all_reduce.side_effect = mock_all_reduce_func

        # Call the function
        result = reduce_tensor(input_tensor)

        # Assertions
        expected_result = torch.tensor([1.0, 2.0, 3.0])  # (4 * input) / 4
        torch.testing.assert_close(result, expected_result)

        # Check if all_reduce was called
        mock_all_reduce.assert_called_once()

        # Check if get_world_size was called
        mock_get_world_size.assert_called_once()


class TestLoadPretrained(unittest.TestCase):
    """
    Test suite for the load_pretrained function.

    This test covers:
    1. Loading a pre-trained model
    2. Handling of encoder prefix in checkpoint keys
    3. Remapping keys for SWIN models
    4. Loading state dict into the model
    """

    def setUp(self):
        self.config = Mock()
        self.model = Mock()
        self.logger = Mock()

    @patch('torch.load')
    @patch('torch.cuda.empty_cache')
    @patch('pytorch_caney.training.mim_utils.remap_pretrained_keys_swin')
    def test_load_pretrained(self, mock_remap, mock_empty_cache, mock_load):
        # Setup
        self.config.MODEL.PRETRAINED = 'pretrained_model.pth'
        self.config.MODEL.TYPE = 'swin'

        # Mock torch.load
        mock_load.return_value = {
            'model': {
                'encoder.layer1': torch.randn(3, 3),
                'encoder.layer2': torch.randn(3, 3),
                'other_key': torch.randn(3, 3)
            }
        }

        # Mock remap_pretrained_keys_swin
        mock_remap.return_value = {
            'layer1': torch.randn(3, 3),
            'layer2': torch.randn(3, 3)
        }

        # Mock model.load_state_dict
        self.model.load_state_dict.return_value = Mock()

        # Call the function
        load_pretrained(self.config, self.model, self.logger)

        # Assertions
        mock_load.assert_called_once_with('pretrained_model.pth', map_location='cpu')
        mock_remap.assert_called_once()
        self.model.load_state_dict.assert_called_once()
        mock_empty_cache.assert_called_once()

        # Check logger calls
        self.logger.info.assert_any_call(">>>>>>>>>> Fine-tuned from pretrained_model.pth ..........")
        self.logger.info.assert_any_call('Detect pre-trained model, remove [encoder.] prefix.')
        self.logger.info.assert_any_call(">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        self.logger.info.assert_any_call(">>>>>>>>>> loaded successfully 'pretrained_model.pth'")

    @patch('torch.load')
    def test_load_pretrained_non_encoder(self, mock_load):
        # Setup for a non-encoder model
        self.config.MODEL.PRETRAINED = 'non_encoder_model.pth'
        self.config.MODEL.TYPE = 'swin'

        mock_load.return_value = {
            'model': {
                'layer1': torch.randn(3, 3),
                'layer2': torch.randn(3, 3)
            }
        }

        # Call the function
        load_pretrained(self.config, self.model, self.logger)

        # Check logger calls
        self.logger.info.assert_any_call('Detect non-pre-trained model, pass without doing anything.')

    @patch('torch.load')
    def test_load_pretrained_unsupported_model(self, mock_load):
        # Setup for an unsupported model type
        self.config.MODEL.PRETRAINED = 'unsupported_model.pth'
        self.config.MODEL.TYPE = 'unsupported'

        mock_load.return_value = {'model': {}}

        # Check that NotImplementedError is raised
        with self.assertRaises(NotImplementedError):
            load_pretrained(self.config, self.model, self.logger)


class TestRemapPretrainedKeysSwin(unittest.TestCase):
    """
    Test suite for the remap_pretrained_keys_swin function.

    This test covers:
    1. Geometric interpolation for mismatched patch sizes
    2. Handling of relative position bias tables
    3. Removal of specific keys from the checkpoint model
    """

    def setUp(self):
        self.model = Mock()
        self.checkpoint_model = {}
        self.logger = Mock()

    @patch('numpy.arange')
    @patch('scipy.interpolate.interp2d')
    def test_remap_pretrained_keys_swin(self, mock_interp2d, mock_arange):
        # Setup mock model state dict
        self.model.state_dict.return_value = {
            "layers.0.blocks.0.attn.relative_position_bias_table": torch.randn(49, 3)
        }

        # Setup mock checkpoint model
        self.checkpoint_model = {
            "layers.0.blocks.0.attn.relative_position_bias_table": torch.randn(25, 3),
            "layers.0.blocks.0.attn.relative_position_index": torch.randn(49),
            "layers.0.blocks.0.attn.relative_coords_table": torch.randn(49, 2),
            "layers.0.blocks.0.attn.attn_mask": torch.randn(1, 1, 49, 49)
        }

        # Mock interpolation
        mock_interp2d.return_value = lambda x, y: np.random.rand(7, 7)
        mock_arange.return_value = np.array([-3, -2, -1, 0, 1, 2, 3])

        # Call the function
        result = remap_pretrained_keys_swin(self.model, self.checkpoint_model, self.logger)

        # Assertions
        self.assertIn("layers.0.blocks.0.attn.relative_position_bias_table", result)
        self.assertEqual(result["layers.0.blocks.0.attn.relative_position_bias_table"].shape, (49, 3))
        
        # Check if specific keys are removed
        self.assertNotIn("layers.0.blocks.0.attn.relative_position_index", result)
        self.assertNotIn("layers.0.blocks.0.attn.relative_coords_table", result)
        self.assertNotIn("layers.0.blocks.0.attn.attn_mask", result)

        # Verify logger calls
        self.logger.info.assert_called()

    def test_mismatched_heads(self):
        # Setup for mismatched number of heads
        self.model.state_dict.return_value = {
            "layers.0.blocks.0.attn.relative_position_bias_table": torch.randn(49, 4)
        }
        self.checkpoint_model = {
            "layers.0.blocks.0.attn.relative_position_bias_table": torch.randn(49, 3)
        }

        result = remap_pretrained_keys_swin(self.model, self.checkpoint_model, self.logger)

        # The original tensor should remain unchanged
        self.assertTrue(torch.equal(result["layers.0.blocks.0.attn.relative_position_bias_table"], 
                                    self.checkpoint_model["layers.0.blocks.0.attn.relative_position_bias_table"]))

        # Verify logger calls
        self.logger.info.assert_called_with("Error in loading layers.0.blocks.0.attn.relative_position_bias_table, passing......")


class TestRemapPretrainedKeysVit(unittest.TestCase):
    """
    Test suite for the remap_pretrained_keys_vit function.

    This test covers:
    1. Handling of relative position bias
    2. Geometric interpolation for mismatched patch sizes
    3. Proper key remapping in the checkpoint model
    """

    def setUp(self):
        self.model = Mock()
        self.checkpoint_model = {}
        self.logger = Mock()

    @patch('torch.Tensor')
    @patch('numpy.arange')
    @patch('scipy.interpolate.interp2d')
    def test_remap_pretrained_keys(self, mock_interp2d, mock_arange, mock_tensor):
        # Setup mock model
        self.model.use_rel_pos_bias = True
        self.model.get_num_layers.return_value = 2
        self.model.patch_embed.patch_shape = [16, 16]
        self.model.state_dict.return_value = {
            "blocks.0.attn.relative_position_bias_table": torch.randn(49, 3),
            "blocks.1.attn.relative_position_bias_table": torch.randn(49, 3)
        }

        # Setup mock checkpoint model
        self.checkpoint_model = {
            "rel_pos_bias.relative_position_bias_table": torch.randn(49, 3),
            "blocks.0.attn.relative_position_bias_table": torch.randn(25, 3),
            "blocks.1.attn.relative_position_bias_table": torch.randn(25, 3),
            "blocks.0.attn.relative_position_index": torch.randn(49),
            "blocks.1.attn.relative_position_index": torch.randn(49)
        }

        # Mock interpolation
        mock_interp2d.return_value = lambda x, y: np.random.rand(7, 7)
        mock_arange.return_value = np.array([-3, -2, -1, 0, 1, 2, 3])
        mock_tensor.return_value = torch.randn(49, 1)

        # Call the function
        result = remap_pretrained_keys_vit(self.model, self.checkpoint_model, self.logger)

        # Assertions
        self.assertNotIn("rel_pos_bias.relative_position_bias_table", result)
        self.assertNotIn("blocks.0.attn.relative_position_index", result)
        self.assertNotIn("blocks.1.attn.relative_position_index", result)
        self.assertIn("blocks.0.attn.relative_position_bias_table", result)
        self.assertIn("blocks.1.attn.relative_position_bias_table", result)
        self.assertEqual(result["blocks.0.attn.relative_position_bias_table"].shape, (49, 3))
        self.assertEqual(result["blocks.1.attn.relative_position_bias_table"].shape, (49, 3))

        # Verify logger calls
        self.logger.info.assert_called()
