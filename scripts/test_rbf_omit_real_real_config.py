"""CPU Hydra/critic wiring checks for omission of the RBF real-real term."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mmd_reward
from test_rbf_cross_sentence import critic_getter


def config(**overrides):
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        cfg = compose(config_name='config', overrides=[
            'algo=mdlm', 'adversarial_distill.mmd_reward_type=feature',
            'adversarial_distill.mmd_feature_source=diffusion',
            'adversarial_distill.mmd_kernel=rbf',
            'adversarial_distill.mmd_kernel_source=draws',
            'adversarial_distill.mmd_kernel_level=token',
            'adversarial_distill.mmd_kernel_batch_size=1',
            'adversarial_distill.mmd_rbf_omit_real_real=true'])
    return OmegaConf.merge(cfg, {'adversarial_distill': overrides})


class OmitRealConfigTests(unittest.TestCase):
    def make_critic(self, cfg):
        teacher = object()
        stub = SimpleNamespace(layer_description='CPU fixture')
        with mock.patch.object(mmd_reward, 'FrozenBackboneFeatures', return_value=stub) as factory:
            with redirect_stdout(io.StringIO()):
                actual = critic_getter()(SimpleNamespace(config=cfg, teacher=teacher, mask_index=12))
        self.assertIs(actual, stub)
        self.assertIs(factory.call_args.args[0], teacher)
        return actual

    def test_hydra_default_is_disabled(self):
        cfg = OmegaConf.load(ROOT / 'configs/config.yaml')
        self.assertIs(cfg.adversarial_distill.mmd_rbf_omit_real_real, False)
        critic = self.make_critic(config(mmd_rbf_omit_real_real=False))
        self.assertFalse(critic.rbf_omit_real_real)

    def test_flag_reaches_critic_for_both_levels_sources_and_distances(self):
        for source in ('rows', 'draws'):
            for level in ('token', 'sentence'):
                for distance in ('squared_euclidean', 'pseudo_huber'):
                    with self.subTest(source=source, level=level, distance=distance):
                        critic = self.make_critic(config(
                            mmd_kernel_source=source, mmd_kernel_level=level,
                            mmd_kernel_batch_size=2, mmd_rbf_distance=distance))
                        self.assertTrue(critic.rbf_omit_real_real)
                        self.assertFalse(critic.rbf_exclude_same_sentence)
                        self.assertEqual(critic.rbf_distance, distance)

    def test_token_k1_and_forward_chunking_are_accepted(self):
        critic = self.make_critic(config(mmd_forward_batch_size=8))
        self.assertTrue(critic.rbf_omit_real_real)
        self.assertEqual(critic.kernel_batch_size, 1)

    def test_existing_cross_sentence_flag_can_be_combined(self):
        critic = self.make_critic(config(mmd_kernel_batch_size=2,
                                         mmd_rbf_exclude_same_sentence=True))
        self.assertTrue(critic.rbf_omit_real_real)
        self.assertTrue(critic.rbf_exclude_same_sentence)

    def test_non_rbf_and_ratio_are_rejected_before_feature_loading(self):
        for options in (dict(mmd_kernel='linear'),
                        dict(mmd_kernel='pseudo_huber', mmd_kernel_level='sentence'),
                        dict(mmd_reward_type='idlm')):
            with self.subTest(options=options), mock.patch.object(mmd_reward, 'FrozenBackboneFeatures') as factory:
                with self.assertRaisesRegex(ValueError, 'mmd_rbf_omit_real_real'):
                    critic_getter()(SimpleNamespace(config=config(**options),
                                                      teacher=object(), mask_index=12))
                factory.assert_not_called()

    def test_feature_mse_ignores_kernel_only_flag(self):
        critic = self.make_critic(config(mmd_reward_type='feature_mse'))
        self.assertFalse(critic.rbf_omit_real_real)
        self.assertEqual(critic.reward_type, 'feature_mse')


if __name__ == '__main__':
    unittest.main(verbosity=2)
