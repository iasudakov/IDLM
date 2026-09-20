"""CPU Hydra/critic wiring checks for omission of the RBF fake-fake term."""
from contextlib import redirect_stdout
import io
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

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
            'adversarial_distill.mmd_rbf_omit_fake_fake=true'])
    return OmegaConf.merge(cfg, {'adversarial_distill': overrides})


class OmitFakeConfigTests(unittest.TestCase):
    def make_critic(self, cfg):
        teacher = object()
        stub = SimpleNamespace(layer_description='CPU fixture')
        with mock.patch.object(mmd_reward, 'FrozenBackboneFeatures', return_value=stub) as factory:
            with redirect_stdout(io.StringIO()):
                actual = critic_getter()(SimpleNamespace(config=cfg, teacher=teacher, mask_index=12))
        self.assertIs(actual, stub)
        self.assertIs(factory.call_args.args[0], teacher)
        return actual

    def test_default_is_disabled_and_legacy_config_is_accepted(self):
        cfg = OmegaConf.load(ROOT / 'configs/config.yaml')
        self.assertIs(cfg.adversarial_distill.mmd_rbf_omit_fake_fake, False)
        legacy = config(mmd_rbf_omit_fake_fake=False)
        with open_dict(legacy.adversarial_distill):
            del legacy.adversarial_distill['mmd_rbf_omit_fake_fake']
        critic = self.make_critic(legacy)
        self.assertFalse(critic.rbf_omit_fake_fake)
        self.assertFalse(critic.rbf_omit_real_real)

    def test_both_flags_are_independent_across_all_rbf_modes(self):
        for source, level, distance, omit_real in itertools.product(
                ('rows', 'draws'), ('token', 'sentence'),
                ('squared_euclidean', 'pseudo_huber'), (False, True)):
            with self.subTest(source=source, level=level, distance=distance, omit_real=omit_real):
                critic = self.make_critic(config(
                    mmd_kernel_source=source, mmd_kernel_level=level,
                    mmd_kernel_batch_size=2, mmd_rbf_distance=distance,
                    mmd_rbf_omit_real_real=omit_real))
                self.assertTrue(critic.rbf_omit_fake_fake)
                self.assertEqual(critic.rbf_omit_real_real, omit_real)
                self.assertEqual(critic.rbf_distance, distance)

    def test_k1_chunking_and_cross_sentence_combination(self):
        critic = self.make_critic(config(mmd_forward_batch_size=8))
        self.assertTrue(critic.rbf_omit_fake_fake)
        self.assertEqual(critic.kernel_batch_size, 1)
        critic = self.make_critic(config(
            mmd_kernel_batch_size=2, mmd_rbf_exclude_same_sentence=True))
        self.assertTrue(critic.rbf_omit_fake_fake)
        self.assertTrue(critic.rbf_exclude_same_sentence)
        with self.assertRaisesRegex(ValueError, 'mmd_kernel_batch_size >= 2'):
            self.make_critic(config(mmd_rbf_exclude_same_sentence=True))

    def test_non_rbf_and_ratio_rejected_before_feature_loading(self):
        for options in (dict(mmd_kernel='linear'),
                        dict(mmd_kernel='pseudo_huber', mmd_kernel_level='sentence'),
                        dict(mmd_reward_type='idlm')):
            with self.subTest(options=options):
                with mock.patch.object(mmd_reward, 'FrozenBackboneFeatures') as factory:
                    with self.assertRaisesRegex(ValueError, 'mmd_rbf_omit_fake_fake'):
                        critic_getter()(SimpleNamespace(config=config(**options),
                                                       teacher=object(), mask_index=12))
                    factory.assert_not_called()

    def test_feature_mse_ignores_both_omission_flags(self):
        critic = self.make_critic(config(
            mmd_reward_type='feature_mse', mmd_rbf_omit_real_real=True))
        self.assertFalse(critic.rbf_omit_fake_fake)
        self.assertFalse(critic.rbf_omit_real_real)
        self.assertEqual(critic.reward_type, 'feature_mse')


if __name__ == '__main__':
    unittest.main(verbosity=2)
