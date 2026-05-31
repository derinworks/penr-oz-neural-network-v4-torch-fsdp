import os
import unittest
from unittest.mock import patch, MagicMock
import torch
import fsdp
from neural_net_model import NeuralNetworkModel
from mappers import Mapper


class TestFSDP(unittest.TestCase):

    def test_is_dist_false(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(fsdp.is_dist())

    def test_is_dist_true(self):
        with patch.dict(os.environ, {"RANK": "0"}):
            self.assertTrue(fsdp.is_dist())

    def test_dist_rank_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(fsdp.dist_rank(), 0)

    def test_dist_rank_set(self):
        with patch.dict(os.environ, {"RANK": "2"}):
            self.assertEqual(fsdp.dist_rank(), 2)

    def test_dist_local_rank_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(fsdp.dist_local_rank(), 0)

    def test_dist_local_rank_set(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            self.assertEqual(fsdp.dist_local_rank(), 3)

    def test_dist_world_size_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(fsdp.dist_world_size(), 1)

    def test_dist_world_size_set(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            self.assertEqual(fsdp.dist_world_size(), 4)

    def test_master_proc_true(self):
        with patch.dict(os.environ, {"RANK": "0"}):
            self.assertTrue(fsdp.master_proc())

    def test_master_proc_false(self):
        with patch.dict(os.environ, {"RANK": "1"}):
            self.assertFalse(fsdp.master_proc())

    def test_master_proc_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(fsdp.master_proc())

    @patch('fsdp.cuda.device_count')
    @patch('fsdp.elastic_launch')
    @patch('fsdp.running_on_linux', return_value=False)
    @patch('fsdp.detect_active_ip_family', return_value="ipv4")
    def test_launch_single_node_cuda(self, mock_detect_active_ip_family, mock_running_on_linux, mock_elastic_launch, mock_device_count):
        mock_device_count.return_value = 2
        mock_worker = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            fsdp.launch_single_node("test_run", "cuda", mock_worker, "arg1", "arg2")

            self.assertTrue(mock_elastic_launch.called)
            call_args = mock_elastic_launch.call_args
            config = call_args[0][0]
            self.assertEqual(config.nproc_per_node, 2)
            self.assertEqual(config.run_id, "test_run")
            # LaunchConfig may normalize the endpoint into bracketed host:port form.
            self.assertIn(config.rdzv_endpoint, {"127.0.0.1:0", "[127.0.0.1]:0"})
            self.assertEqual(config.local_addr, "127.0.0.1")
            self.assertEqual(os.environ.get("MASTER_ADDR"), "127.0.0.1")

    @patch('fsdp.cuda.device_count')
    @patch('fsdp.elastic_launch')
    @patch('fsdp.running_on_linux', return_value=False)
    @patch('fsdp.detect_active_ip_family', return_value="ipv6")
    def test_launch_single_node_cuda_ipv6(self, mock_detect_active_ip_family, mock_running_on_linux, mock_elastic_launch, mock_device_count):
        mock_device_count.return_value = 2
        mock_worker = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            fsdp.launch_single_node("test_run", "cuda", mock_worker, "arg1", "arg2")

            call_args = mock_elastic_launch.call_args
            config = call_args[0][0]
            self.assertEqual(config.nproc_per_node, 2)
            self.assertEqual(config.run_id, "test_run")
            self.assertEqual(config.rdzv_endpoint, "[::1]:0")
            self.assertEqual(config.local_addr, "::1")
            self.assertEqual(os.environ.get("MASTER_ADDR"), "::1")
            self.assertEqual(os.environ.get("GLOO_USE_IPV6"), "1")

    @patch('fsdp.cpu_count')
    @patch('fsdp.elastic_launch')
    def test_launch_single_node_cpu(self, mock_elastic_launch, mock_cpu_count):
        mock_cpu_count.return_value = 8
        mock_worker = MagicMock()

        fsdp.launch_single_node("test_run", "cpu", mock_worker, "arg1")

        self.assertTrue(mock_elastic_launch.called)
        call_args = mock_elastic_launch.call_args
        config = call_args[0][0]
        self.assertEqual(config.nproc_per_node, 4)  # max(1, 8 // 2)

    @patch('fsdp.mps.device_count')
    @patch('fsdp.elastic_launch')
    def test_launch_single_node_mps(self, mock_elastic_launch, mock_device_count):
        mock_device_count.return_value = 1
        mock_worker = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            fsdp.launch_single_node("test_run", "mps", mock_worker, "arg1")

            self.assertTrue(mock_elastic_launch.called)
            call_args = mock_elastic_launch.call_args
            config = call_args[0][0]
            self.assertEqual(config.nproc_per_node, 1)
            self.assertEqual(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "1")

    @patch('fsdp.get_backend')
    @patch('fsdp.all_reduce')
    def test_dist_all_reduce_nccl(self, mock_all_reduce, mock_get_backend):
        mock_get_backend.return_value = 'nccl'
        tensor = torch.tensor([1.0, 2.0, 3.0])

        fsdp.dist_all_reduce(tensor)

        self.assertTrue(mock_all_reduce.called)

    @patch('fsdp.get_backend')
    @patch('fsdp.all_reduce')
    def test_dist_all_reduce_gloo(self, mock_all_reduce, mock_get_backend):
        mock_get_backend.return_value = 'gloo'
        tensor = torch.tensor([1.0, 2.0, 3.0])

        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            fsdp.dist_all_reduce(tensor)

        self.assertTrue(mock_all_reduce.called)

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"version": 1}')
    @patch('logging.config.dictConfig')
    def test_reconfig_logging(self, mock_dict_config, mock_open):
        with patch.dict(os.environ, {}, clear=True):
            fsdp.reconfig_logging()

        self.assertTrue(mock_open.called)
        self.assertTrue(mock_dict_config.called)

    @patch('fsdp.Path.mkdir')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"version": 1, "handlers": {}, "root": {"handlers": []}}')
    @patch('logging.config.dictConfig')
    @patch('fsdp.running_on_linux', return_value=False)
    def test_reconfig_logging_dist_adds_rank_file_handler(self, mock_running_on_linux, mock_dict_config, mock_open, mock_mkdir):
        with patch.dict(os.environ, {"RANK": "1"}, clear=True):
            fsdp.reconfig_logging()

        self.assertTrue(mock_open.called)
        self.assertTrue(mock_mkdir.called)
        cfg = mock_dict_config.call_args[0][0]
        self.assertIn("dist_file", cfg["handlers"])
        self.assertEqual(cfg["handlers"]["dist_file"]["filename"], "logs/dist_rank01.log")
        self.assertIn("dist_file", cfg["root"]["handlers"])

    def test_use_fsdp_false_when_not_dist(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(fsdp.use_fsdp('cpu'))

    def test_use_fsdp_true_for_cuda(self):
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"}):
            self.assertTrue(fsdp.use_fsdp('cuda'))

    def test_use_fsdp_true_for_cpu(self):
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "4"}):
            self.assertTrue(fsdp.use_fsdp('cpu'))

    def test_use_fsdp_false_for_mps_single_process(self):
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"}):
            self.assertFalse(fsdp.use_fsdp('mps'))

    def test_use_fsdp_true_for_mps_multi_process(self):
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"}):
            self.assertTrue(fsdp.use_fsdp('mps'))

    def test_mixed_precision_policy_none_for_cpu(self):
        self.assertIsNone(fsdp.mixed_precision_policy('cpu'))
        self.assertIsNone(fsdp.mixed_precision_policy(torch.device('cpu')))

    @patch('fsdp.cuda.is_bf16_supported', return_value=True)
    def test_mixed_precision_policy_cuda_bfloat16(self, mock_bf16):
        policy = fsdp.mixed_precision_policy('cuda')
        self.assertIsNotNone(policy)
        self.assertEqual(policy.param_dtype, torch.bfloat16)
        self.assertEqual(policy.reduce_dtype, torch.bfloat16)

    @patch('fsdp.cuda.is_bf16_supported', return_value=False)
    def test_mixed_precision_policy_cuda_float16(self, mock_bf16):
        policy = fsdp.mixed_precision_policy(torch.device('cuda'))
        self.assertIsNotNone(policy)
        self.assertEqual(policy.param_dtype, torch.float16)
        self.assertEqual(policy.reduce_dtype, torch.float16)

    @patch('fsdp.fully_shard')
    def test_shard_model_wraps_param_layers_and_root(self, mock_fully_shard):
        layers = [{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
                  {"linear": {"in_features": 2, "out_features": 8}},
                  {"softmaxlast": {"dim": -1}}]
        model = NeuralNetworkModel("test-shard", Mapper(layers, {"sgd": {"lr": .01}}))
        n_param_layers = sum(1 for layer in model.layers
                             if next(layer.parameters(), None) is not None)

        result = fsdp.shard_model(model, torch.device('cpu'))

        # Same instance returned, wrapped once per param-bearing layer + root.
        self.assertIs(result, model)
        self.assertEqual(mock_fully_shard.call_count, n_param_layers + 1)
        # CPU device -> no mixed precision policy is applied.
        for call in mock_fully_shard.call_args_list:
            self.assertNotIn('mp_policy', call.kwargs)

    @patch('fsdp.fully_shard')
    @patch('fsdp.mixed_precision_policy')
    def test_shard_model_applies_mixed_precision_policy(self, mock_mp_policy, mock_fully_shard):
        sentinel = MagicMock()
        mock_mp_policy.return_value = sentinel
        layers = [{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
                  {"linear": {"in_features": 2, "out_features": 8}},
                  {"softmaxlast": {"dim": -1}}]
        model = NeuralNetworkModel("test-shard-mp", Mapper(layers, {"sgd": {"lr": .01}}))

        fsdp.shard_model(model, "cuda")

        self.assertTrue(mock_fully_shard.called)
        for call in mock_fully_shard.call_args_list:
            self.assertIs(call.kwargs.get('mp_policy'), sentinel)


if __name__ == '__main__':
    unittest.main()
