"""Exercise the actual startup hook offline, without constructing Lightning."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Config(SimpleNamespace):
  def get(self, key, default=None):
    return getattr(self, key, default)


class Parent:
  def on_train_start(self):
    self.events.append('parent')


def startup_class():
  tree = ast.parse((ROOT / 'algo.py').read_text())
  mdlm = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MDLM')
  method = next(n for n in mdlm.body if isinstance(n, ast.FunctionDef) and n.name == 'on_train_start')
  cls = ast.parse('class StartupProbe(Parent):\n    pass').body[0]
  cls.body = [method]
  namespace = {'Parent': Parent, 'utils': SimpleNamespace(
    activate_model=lambda model: None, freeze_model=lambda model: None)}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
               'algo.py', 'exec'), namespace)
  return namespace['StartupProbe']


class SDTTStartupTests(unittest.TestCase):
  def make(self, **overrides):
    settings = dict(mmd_feature_source='sdtt_mdlm', mmd_reward_weight=1., mmd_reward_type='feature')
    settings.update(overrides)

    class Probe(startup_class()):
      @property
      def _mmd_critic(self):
        self.events.append('feature model')
        if self.download_error:
          raise self.download_error
        return object()

    model = Probe()
    model.is_adversarial_distill = True
    model.config = SimpleNamespace(adversarial_distill=Config(**settings))
    model.ema_fake = None
    model.backbone = object()
    model.fake_model = object()
    model.events = []
    model.download_error = None
    model._needs_fake_phase = lambda: False
    model._sync_noise_schedule = lambda: model.events.append('noise schedule')
    return model

  def test_prepares_features_after_parent_before_training(self):
    model = self.make()
    model.on_train_start()
    self.assertEqual(model.events, ['noise schedule', 'parent', 'feature model'])

  def test_feature_mse_also_prepares_model(self):
    model = self.make(mmd_reward_type='feature_mse')
    model.on_train_start()
    self.assertEqual(model.events[-1], 'feature model')

  def test_legacy_sources_do_not_prepare_sdtt(self):
    for source in ('ar', 'diffusion'):
      with self.subTest(source=source):
        model = self.make(mmd_feature_source=source)
        model.on_train_start()
        self.assertNotIn('feature model', model.events)

  def test_disabled_reward_does_not_download(self):
    model = self.make(mmd_reward_weight=0.)
    model.on_train_start()
    self.assertNotIn('feature model', model.events)

  def test_ratio_reward_does_not_download_unused_features(self):
    model = self.make(mmd_reward_type='idlm')
    model.on_train_start()
    self.assertNotIn('feature model', model.events)

  def test_non_distillation_does_not_download(self):
    model = self.make()
    model.is_adversarial_distill = False
    model.on_train_start()
    self.assertEqual(model.events, ['parent'])

  def test_download_failure_stops_startup(self):
    model = self.make()
    model.download_error = RuntimeError('Hub unavailable')
    with self.assertRaisesRegex(RuntimeError, 'Hub unavailable'):
      model.on_train_start()
    self.assertEqual(model.events, ['noise schedule', 'parent', 'feature model'])


if __name__ == '__main__':
  unittest.main(verbosity=2)
