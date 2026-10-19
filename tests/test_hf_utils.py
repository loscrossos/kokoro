"""
Unit tests for centralized checkpoint resolution and entry script parameterization.
"""

import ast
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from kokoro.hf_utils import (
    find_repo_root,
    get_checkpoints_dir,
    set_checkpoints_dir,
    reset_checkpoints_dir,
    ensure_checkpoints_dir,
    parse_repo_id,
    get_repo_name,
    resolve_model_path,
    resolve_pretrained_path,
    load_custom_model,
)


class TestRepoIdParsing(unittest.TestCase):
    """Verifies URL and repo ID parsing across various formats."""

    def test_standard_repo_id(self):
        author, repo, subpath = parse_repo_id("hexgrad/Kokoro-82M")
        self.assertEqual(author, "hexgrad")
        self.assertEqual(repo, "Kokoro-82M")
        self.assertIsNone(subpath)
        self.assertEqual(get_repo_name("hexgrad/Kokoro-82M"), "Kokoro-82M")

    def test_naked_repo_id(self):
        author, repo, subpath = parse_repo_id("Kokoro-82M")
        self.assertIsNone(author)
        self.assertEqual(repo, "Kokoro-82M")
        self.assertIsNone(subpath)
        self.assertEqual(get_repo_name("Kokoro-82M"), "Kokoro-82M")

    def test_huggingface_repo_url(self):
        url = "https://huggingface.co/hexgrad/Kokoro-82M"
        author, repo, subpath = parse_repo_id(url)
        self.assertEqual(author, "hexgrad")
        self.assertEqual(repo, "Kokoro-82M")
        self.assertIsNone(subpath)
        self.assertEqual(get_repo_name(url), "Kokoro-82M")

    def test_huggingface_git_url(self):
        url = "https://huggingface.co/hexgrad/Kokoro-82M.git"
        author, repo, subpath = parse_repo_id(url)
        self.assertEqual(author, "hexgrad")
        self.assertEqual(repo, "Kokoro-82M")
        self.assertEqual(get_repo_name(url), "Kokoro-82M")

    def test_huggingface_resolve_file_url(self):
        url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth"
        author, repo, subpath = parse_repo_id(url)
        self.assertEqual(author, "hexgrad")
        self.assertEqual(repo, "Kokoro-82M")
        self.assertEqual(subpath, "kokoro-v1_0.pth")

    def test_huggingface_resolve_subpath_url(self):
        url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt"
        author, repo, subpath = parse_repo_id(url)
        self.assertEqual(author, "hexgrad")
        self.assertEqual(repo, "Kokoro-82M")
        self.assertEqual(subpath, "voices/af_heart.pt")

    def test_huggingface_raw_url(self):
        url = "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json"
        author, repo, subpath = parse_repo_id(url)
        self.assertEqual(author, "hexgrad")
        self.assertEqual(repo, "Kokoro-82M")
        self.assertEqual(subpath, "config.json")

    def test_modelscope_url(self):
        url = "https://modelscope.cn/models/hexgrad/Kokoro-82M"
        author, repo, subpath = parse_repo_id(url)
        self.assertEqual(author, "hexgrad")
        self.assertEqual(repo, "Kokoro-82M")
        self.assertIsNone(subpath)
        self.assertEqual(get_repo_name(url), "Kokoro-82M")

    def test_modelscope_file_url(self):
        url = "https://modelscope.cn/models/author/reponame/resolve/master/weights.bin"
        author, repo, subpath = parse_repo_id(url)
        self.assertEqual(author, "author")
        self.assertEqual(repo, "reponame")
        self.assertEqual(subpath, "weights.bin")

    def test_generic_author_reponame_subdir(self):
        url = "https://huggingface.co/author/reponame/subdir/model.ext"
        author, repo, subpath = parse_repo_id(url)
        self.assertEqual(author, "author")
        self.assertEqual(repo, "reponame")
        self.assertEqual(subpath, "subdir/model.ext")
        self.assertEqual(get_repo_name(url), "reponame")


class TestCheckpointsDirManagement(unittest.TestCase):
    """Verifies directory synchronization, overrides, and environment variables."""

    def setUp(self):
        reset_checkpoints_dir()
        self.original_env = os.environ.get("CHECKPOINT_DIR")
        self.original_hf_cache = os.environ.get("HF_HUB_CACHE")
        self.original_hf_home = os.environ.get("HF_HOME")

    def tearDown(self):
        reset_checkpoints_dir()
        if self.original_env is not None:
            os.environ["CHECKPOINT_DIR"] = self.original_env
        else:
            os.environ.pop("CHECKPOINT_DIR", None)

        if self.original_hf_cache is not None:
            os.environ["HF_HUB_CACHE"] = self.original_hf_cache
        else:
            os.environ.pop("HF_HUB_CACHE", None)

        if self.original_hf_home is not None:
            os.environ["HF_HOME"] = self.original_hf_home
        else:
            os.environ.pop("HF_HOME", None)

    def test_default_checkpoints_dir(self):
        os.environ.pop("CHECKPOINT_DIR", None)
        ckpt_dir = get_checkpoints_dir()
        self.assertTrue(str(ckpt_dir).endswith("checkpoints"))

    def test_environment_variable_override(self):
        custom = Path("D:/test_ckpts").resolve()
        os.environ["CHECKPOINT_DIR"] = str(custom)
        ckpt_dir = get_checkpoints_dir()
        self.assertEqual(ckpt_dir, custom)

    def test_set_checkpoints_dir_synchronization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "my_checkpoints"
            ret = set_checkpoints_dir(target)
            self.assertEqual(ret, target.resolve())
            self.assertEqual(get_checkpoints_dir(), target.resolve())
            self.assertEqual(os.environ["CHECKPOINT_DIR"], str(target.resolve()))
            self.assertEqual(os.environ["HF_HUB_CACHE"], str(target.resolve() / "hub"))
            self.assertEqual(os.environ["HF_HOME"], str(target.resolve() / "hf_home"))


class TestPathResolution(unittest.TestCase):
    """Verifies local path resolution, fallback order, and custom base_dir overrides."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_canonical_repo_subfolder_resolution(self):
        # checkpoints/<reponame>/<filename>
        model_file = self.base / "Kokoro-82M" / "kokoro-v1_0.pth"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.touch()

        resolved = resolve_model_path(
            repo_id="hexgrad/Kokoro-82M",
            filename="kokoro-v1_0.pth",
            base_dir=self.base,
            must_exist=True
        )
        self.assertEqual(resolved, model_file.resolve())

    def test_canonical_subpath_resolution(self):
        # checkpoints/<reponame>/voices/af_heart.pt
        voice_file = self.base / "Kokoro-82M" / "voices" / "af_heart.pt"
        voice_file.parent.mkdir(parents=True, exist_ok=True)
        voice_file.touch()

        resolved = resolve_model_path(
            repo_id="hexgrad/Kokoro-82M",
            filename="voices/af_heart.pt",
            base_dir=self.base,
            must_exist=True
        )
        self.assertEqual(resolved, voice_file.resolve())

    def test_flat_fallback_resolution(self):
        # checkpoints/kokoro-v1_0.pth (flat fallback)
        model_file = self.base / "kokoro-v1_0.pth"
        model_file.touch()

        resolved = resolve_model_path(
            repo_id="hexgrad/Kokoro-82M",
            filename="kokoro-v1_0.pth",
            base_dir=self.base,
            must_exist=True
        )
        self.assertEqual(resolved, model_file.resolve())

    def test_author_reponame_fallback_resolution(self):
        # checkpoints/hexgrad/Kokoro-82M/kokoro-v1_0.pth
        model_file = self.base / "hexgrad" / "Kokoro-82M" / "kokoro-v1_0.pth"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.touch()

        resolved = resolve_model_path(
            repo_id="hexgrad/Kokoro-82M",
            filename="kokoro-v1_0.pth",
            base_dir=self.base,
            must_exist=True
        )
        self.assertEqual(resolved, model_file.resolve())

    def test_direct_file_path(self):
        direct = self.base / "custom_dir" / "my_model.pth"
        direct.parent.mkdir(parents=True, exist_ok=True)
        direct.touch()

        resolved = resolve_model_path(
            repo_id="hexgrad/Kokoro-82M",
            filename=str(direct),
            base_dir=self.base,
            must_exist=True
        )
        self.assertEqual(resolved, direct.resolve())

    def test_nonexistent_must_exist(self):
        resolved = resolve_model_path(
            repo_id="hexgrad/Kokoro-82M",
            filename="absent.pth",
            base_dir=self.base,
            must_exist=True
        )
        self.assertIsNone(resolved)

        # when must_exist=False, returns canonical expected target path
        canonical = resolve_model_path(
            repo_id="hexgrad/Kokoro-82M",
            filename="absent.pth",
            base_dir=self.base,
            must_exist=False
        )
        self.assertEqual(canonical, (self.base / "Kokoro-82M" / "absent.pth").resolve())

    def test_resolve_pretrained_path(self):
        repo_dir = self.base / "Kokoro-82M"
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "config.json").touch()

        resolved = resolve_pretrained_path("hexgrad/Kokoro-82M", base_dir=self.base, must_exist=True)
        self.assertEqual(resolved, repo_dir.resolve())

        # Nonexistent returns model_id when must_exist=True
        absent = resolve_pretrained_path("other/absent-model", base_dir=self.base, must_exist=True)
        self.assertEqual(absent, "other/absent-model")


class TestLoadCustomModel(unittest.TestCase):
    """Verifies load_custom_model offline loading and download fallback."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_offline_immediate_loading(self):
        # Create local file
        target = self.base / "Kokoro-82M" / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"test_key": "local_value"}', encoding="utf-8")

        # Mock hf_hub_download to ensure it is NEVER called when local file exists
        mock_hf = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            result = load_custom_model(
                repo_id="hexgrad/Kokoro-82M",
                filename="config.json",
                base_dir=self.base
            )
            self.assertEqual(result, {"test_key": "local_value"})
            mock_hf.hf_hub_download.assert_not_called()

    def test_download_fallback_when_absent(self):
        expected_dest = self.base / "Kokoro-82M" / "test.json"
        
        def fake_download(*args, **kwargs):
            expected_dest.parent.mkdir(parents=True, exist_ok=True)
            expected_dest.write_text('{"downloaded": true}', encoding="utf-8")
            return str(expected_dest)

        mock_hub = MagicMock()
        mock_hub.hf_hub_download.side_effect = fake_download
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            result = load_custom_model(
                repo_id="hexgrad/Kokoro-82M",
                filename="test.json",
                base_dir=self.base
            )
            self.assertEqual(result, {"downloaded": True})
            self.assertTrue(expected_dest.exists())

    def test_ensure_checkpoints_dir(self):
        target = self.base / "nested" / "ckpts"
        self.assertFalse(target.exists())
        ret = ensure_checkpoints_dir(target)
        self.assertTrue(target.is_dir())
        self.assertEqual(ret, target.resolve())

    def test_download_model_file_helper(self):
        from kokoro.hf_utils import download_model_file
        expected_dest = self.base / "Kokoro-82M" / "weight.pth"
        
        def fake_download(*args, **kwargs):
            expected_dest.parent.mkdir(parents=True, exist_ok=True)
            expected_dest.write_text("weights", encoding="utf-8")
            return str(expected_dest)

        mock_hub = MagicMock()
        mock_hub.hf_hub_download.side_effect = fake_download
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            path = download_model_file("hexgrad/Kokoro-82M", "weight.pth", base_dir=self.base)
            self.assertEqual(path, expected_dest)
            self.assertTrue(expected_dest.exists())


class TestRootShim(unittest.TestCase):
    """Verifies that the root hf_utils shim exports identical functions."""

    def test_root_shim_exports(self):
        import hf_utils
        import kokoro.hf_utils as core_hf_utils

        for attr in hf_utils.__all__:
            self.assertTrue(hasattr(hf_utils, attr), f"Missing {attr} on hf_utils shim")
            self.assertIs(getattr(hf_utils, attr), getattr(core_hf_utils, attr))


class TestGermanLanguageRouting(unittest.TestCase):
    """Verifies PR #371 German language additions and offline voice resolution."""

    def test_german_alias_and_lang_code(self):
        mock_torch = MagicMock()
        mock_misaki = MagicMock()
        mock_hf = MagicMock()
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torch.nn": MagicMock(),
            "transformers": MagicMock(),
            "misaki": mock_misaki,
            "misaki.en": MagicMock(),
            "misaki.espeak": MagicMock(),
            "huggingface_hub": mock_hf,
            "attr": MagicMock(),
        }):
            from kokoro.pipeline import ALIASES, LANG_CODES
            self.assertEqual(ALIASES.get("de"), "d")
            self.assertEqual(LANG_CODES.get("d"), "de")

    def test_missing_german_voice_raises_runtime_error(self):
        mock_torch = MagicMock()
        mock_misaki = MagicMock()
        mock_hf = MagicMock()
        mock_hf.hf_hub_download.side_effect = Exception("Voice not found on Hub")
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torch.nn": MagicMock(),
            "transformers": MagicMock(),
            "misaki": mock_misaki,
            "misaki.en": MagicMock(),
            "misaki.espeak": MagicMock(),
            "huggingface_hub": mock_hf,
            "attr": MagicMock(),
        }):
            from kokoro.pipeline import KPipeline
            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline = KPipeline(lang_code="d", model=False, checkpoint_dir=tmpdir)
                with self.assertRaises(RuntimeError) as ctx:
                    pipeline.load_single_voice("df_kerstin.pt")
                self.assertIn("German voice tensor 'df_kerstin.pt' not found locally", str(ctx.exception))

                # Test non-.pt voice name fallback raising the German-specific error
                with self.assertRaises(RuntimeError) as ctx2:
                    pipeline.load_single_voice("df_kerstin")
                self.assertIn("German voice tensor 'df_kerstin' not found locally", str(ctx2.exception))

    def test_german_community_voice_offline_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            german_voice = base / "kokoro-german-kerstin" / "voices" / "df_kerstin.pt"
            german_voice.parent.mkdir(parents=True, exist_ok=True)
            german_voice.touch()

            resolved = resolve_model_path(
                repo_id="cryptomilk/kokoro-german-kerstin",
                filename="voices/df_kerstin.pt",
                base_dir=base,
                must_exist=True
            )
            self.assertEqual(resolved, german_voice.resolve())


class TestEntryScriptsAST(unittest.TestCase):
    """Verifies via AST parsing that all entry scripts expose --checkpoint-dir."""

    ENTRY_SCRIPTS = [
        "kokoro/__main__.py",
        "demo/app.py",
        "examples/export.py",
        "examples/device_examples.py",
        "examples/phoneme_example.py",
        "examples/make_triton_compatible.py",
        "test_german_inference.py",
    ]

    def test_all_entry_scripts_expose_checkpoint_dir(self):
        repo_root = find_repo_root()
        for rel_path in self.ENTRY_SCRIPTS:
            script_path = repo_root / rel_path
            self.assertTrue(script_path.is_file(), f"Entry script missing: {rel_path}")

            with open(script_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(script_path))

            # Search AST for add_argument calls with '--checkpoint-dir' or string occurrences
            found_arg = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check if call is add_argument
                    func_name = None
                    if isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    elif isinstance(node.func, ast.Name):
                        func_name = node.func.id

                    if func_name == "add_argument":
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and arg.value in ("--checkpoint-dir", "--checkpoint_dir"):
                                found_arg = True
                                break

            self.assertTrue(
                found_arg,
                f"Entry script '{rel_path}' does not expose '--checkpoint-dir' in its ArgumentParser AST"
            )


if __name__ == "__main__":
    unittest.main()
