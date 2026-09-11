"""CPU-only checks for local model bundles: python scripts/test_model_assets.py."""

import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_assets import (
    configure_model_weights,
    pretrained_kwargs,
    resolve_checkpoint,
    resolve_pretrained_path,
)


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ModelAssetsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="model assets test ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle = self.root / "offline bundle"
        self.bundle.mkdir()
        # Resolution must never consult the Hub, including for missing assets.
        for target in ("socket.getaddrinfo", "socket.socket.connect"):
            guard = mock.patch(target, side_effect=AssertionError("network access attempted"))
            guard.start()
            self.addCleanup(guard.stop)

    def asset(self, relative, *, model=True, tokenizer=False, weights="model.safetensors"):
        path = self.bundle / relative
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text('{"model_type": "gpt2"}')
        if model:
            (path / weights).write_bytes(b"test weight placeholder")
        if tokenizer:
            (path / "tokenizer.json").write_text("{}")
            (path / "tokenizer_config.json").write_text('{"tokenizer_class": "GPT2TokenizerFast"}')
        return path.resolve()

    def manifest(self, *assets):
        (self.bundle / "manifest.json").write_text(json.dumps({"schema_version": 1, "assets": assets}))

    def assert_local(self, requested, expected, *, kind="model"):
        self.assertEqual(
            resolve_pretrained_path(requested, str(self.bundle), kind=kind), str(expected)
        )
        self.assertEqual(
            pretrained_kwargs(requested, str(self.bundle), kind=kind),
            {"pretrained_model_name_or_path": str(expected), "local_files_only": True},
        )

    def test_unset_option_preserves_existing_loading_behavior(self):
        requested = "organization/model-name"
        self.assertEqual(pretrained_kwargs(requested), {"pretrained_model_name_or_path": requested})
        self.assertEqual(resolve_pretrained_path(requested), requested)
        self.assertEqual(resolve_checkpoint("checkpoints/not-downloaded.ckpt"), "checkpoints/not-downloaded.ckpt")

    def test_manifest_folder_and_canonical_alias_use_same_assets(self):
        expected = self.asset("gpt2-large", weights="pytorch_model.bin")
        self.manifest({
            "repo_id": "gpt2-large",
            "canonical_repo_id": "openai-community/gpt2-large",
            "folder": "gpt2-large",
            "revision": "bundle-revision",
        })
        for requested in ("gpt2-large", "openai-community/gpt2-large"):
            with self.subTest(requested=requested):
                self.assert_local(requested, expected)

    def test_manifest_without_folder_uses_pinned_cache_snapshot(self):
        expected = self.asset(
            "hf-cache/models--HuggingFaceTB--SmolLM-135M/snapshots/bundled-revision",
            model=False, tokenizer=True,
        )
        self.asset("hf-cache/models--HuggingFaceTB--SmolLM-135M/snapshots/other-revision", model=False, tokenizer=True)
        refs = self.bundle / "hf-cache/models--HuggingFaceTB--SmolLM-135M/refs"
        refs.mkdir()
        (refs / "main").write_text("other-revision")
        self.manifest({
            "repo_id": "HuggingFaceTB/SmolLM-135M",
            "canonical_repo_id": "HuggingFaceTB/SmolLM-135M",
            "revision": "bundled-revision",
            "kind": "tokenizer_only",
        })
        self.assert_local("HuggingFaceTB/SmolLM-135M", expected, kind="tokenizer")

    def test_cache_fallback_obeys_refs_main(self):
        self.asset("hf-cache/models--organization--model/snapshots/aaa-old")
        expected = self.asset("hf-cache/models--organization--model/snapshots/zzz-current")
        refs = self.bundle / "hf-cache/models--organization--model/refs"
        refs.mkdir()
        (refs / "main").write_text("zzz-current\n")
        self.assert_local("organization/model", expected)

    def test_bundle_repo_id_and_basename_directories(self):
        for requested, relative in (
            ("organization/full-name", "organization/full-name"),
            ("organization/short-name", "short-name"),
        ):
            with self.subTest(requested=requested):
                self.assert_local(requested, self.asset(relative))

    def test_explicit_local_directory_with_spaces(self):
        expected = self.asset("saved model")
        self.assert_local(str(expected), expected)
        with working_directory(self.root):
            self.assert_local("offline bundle/saved model", expected)

    def test_tokenizer_only_assets_fail_when_model_is_needed(self):
        expected = self.asset("gpt2-large", model=False, tokenizer=True)
        self.assert_local("gpt2-large", expected, kind="tokenizer")
        with self.assertRaises(FileNotFoundError) as raised:
            pretrained_kwargs("gpt2-large", str(self.bundle), kind="model")
        message = str(raised.exception)
        self.assertIn("gpt2-large", message)
        self.assertTrue("weight" in message.lower() or "model" in message.lower(), message)

    def test_shard_index_requires_every_referenced_weight_file(self):
        expected = self.asset("sharded", model=False)
        index = {"weight_map": {"first": "model-00001-of-00002.safetensors", "second": "model-00002-of-00002.safetensors"}}
        (expected / "model.safetensors.index.json").write_text(json.dumps(index))
        (expected / "model-00001-of-00002.safetensors").write_bytes(b"first shard")
        with self.assertRaises(FileNotFoundError) as raised:
            resolve_pretrained_path("sharded", str(self.bundle))
        self.assertIn("model-00002-of-00002.safetensors", str(raised.exception))
        (expected / "model-00002-of-00002.safetensors").write_bytes(b"second shard")
        self.assert_local("sharded", expected)

    def test_model_requires_config_as_well_as_weights(self):
        path = self.asset("incomplete")
        (path / "config.json").unlink()
        with self.assertRaises(FileNotFoundError) as raised:
            resolve_pretrained_path("incomplete", str(self.bundle))
        self.assertIn("config.json", str(raised.exception))

    def test_missing_model_reports_requested_identity_and_bundle(self):
        with self.assertRaises(FileNotFoundError) as raised:
            pretrained_kwargs("missing-organization/missing-model", str(self.bundle))
        self.assertIn("missing-organization/missing-model", str(raised.exception))
        self.assertIn(str(self.bundle), str(raised.exception))

    def test_checkpoint_uses_launch_directory_before_bundle(self):
        relative = Path("checkpoints/tinygsm/ar_notok.ckpt")
        bundled = self.bundle / relative
        bundled.parent.mkdir(parents=True)
        bundled.write_bytes(b"bundled checkpoint")
        launch = self.root / "launch"
        local = launch / relative
        local.parent.mkdir(parents=True)
        local.write_bytes(b"launch checkpoint")
        with working_directory(launch):
            self.assertEqual(resolve_checkpoint(str(relative), str(self.bundle)), str(local.resolve()))
            self.assertEqual(resolve_checkpoint(str(bundled), str(self.bundle)), str(bundled.resolve()))
            local.unlink()
            self.assertEqual(resolve_checkpoint(str(relative), str(self.bundle)), str(bundled.resolve()))
            with self.assertRaises(FileNotFoundError) as raised:
                resolve_checkpoint("checkpoints/tinygsm/mdlm_notok.ckpt", str(self.bundle))
            self.assertIn("mdlm_notok.ckpt", str(raised.exception))

    def test_config_normalizes_paths_before_working_directory_changes(self):
        from omegaconf import OmegaConf

        launch = self.root / "launch"
        launch.mkdir()
        student = launch / "student.ckpt"
        student.write_bytes(b"student checkpoint")
        teacher = self.bundle / "teacher.ckpt"
        teacher.write_bytes(b"teacher checkpoint")
        for backbone in ("dit", "hf_dit"):
            with self.subTest(backbone=backbone):
                eval_path = "teacher.ckpt" if backbone == "dit" else "organization/model"
                config = OmegaConf.create({
                    "model_weights_dir": "../offline bundle",
                    "algo": {"backbone": backbone},
                    "data": {"tokenizer_name_or_path": "HuggingFaceTB/SmolLM-135M"},
                    "training": {"finetune_path": "student.ckpt"},
                    "eval": {"checkpoint_path": eval_path},
                    "adversarial_distill": {"mmd_ar_model": "teacher.ckpt"},
                })
                with working_directory(self.root):
                    configure_model_weights(config, base_dir=launch)
                self.assertEqual(config.model_weights_dir, str(self.bundle))
                self.assertEqual(config.training.finetune_path, str(student))
                self.assertEqual(config.adversarial_distill.mmd_ar_model, str(teacher))
                self.assertEqual(config.eval.checkpoint_path, str(teacher) if backbone == "dit" else eval_path)
                self.assertEqual(config.data.tokenizer_name_or_path, "HuggingFaceTB/SmolLM-135M")

    def test_missing_checkpoint_can_be_deferred_until_it_is_needed(self):
        requested = "checkpoints/unused.ckpt"
        self.assertEqual(
            resolve_checkpoint(requested, str(self.bundle), required=False), requested
        )
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(requested, str(self.bundle))

    def test_config_does_not_require_inactive_checkpoint_fields(self):
        from omegaconf import OmegaConf

        existing = self.bundle / "existing.ckpt"
        existing.write_bytes(b"active checkpoint")
        for mode in ("train", "sample_eval"):
            with self.subTest(mode=mode):
                training_path = "existing.ckpt" if mode == "train" else "unused-finetune.ckpt"
                eval_path = "unused-eval.ckpt" if mode == "train" else "existing.ckpt"
                config = OmegaConf.create({
                    "mode": mode,
                    "model_weights_dir": str(self.bundle),
                    "algo": {"backbone": "dit"},
                    "training": {"finetune_path": training_path},
                    "eval": {"checkpoint_path": eval_path},
                    "adversarial_distill": {
                        "mmd_reward_weight": 0.0,
                        "mmd_ar_model": "unused-ar.ckpt",
                    },
                })
                configure_model_weights(config, base_dir=self.root)
                self.assertEqual(
                    config.training.finetune_path,
                    str(existing) if mode == "train" else training_path,
                )
                self.assertEqual(
                    config.eval.checkpoint_path,
                    eval_path if mode == "train" else str(existing),
                )
                self.assertEqual(config.adversarial_distill.mmd_ar_model, "unused-ar.ckpt")


if __name__ == "__main__":
    unittest.main()
