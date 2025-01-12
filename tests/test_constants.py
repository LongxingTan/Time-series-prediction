import os
import shutil
import tempfile
import unittest

from tfts.constants import default_assets_cache_path, default_cache_path, default_home


class TestCachePaths(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Save the original environment variables for restoration
        self.original_tfts_home = os.getenv("TFTS_HOME")
        self.original_tfts_hub_cache = os.getenv("TFTS_HUB_CACHE")
        self.original_tfts_assets_cache = os.getenv("TFTS_ASSETS_CACHE")
        self.original_xdg_cache_home = os.getenv("XDG_CACHE_HOME")

    def tearDown(self):
        # Clean up the temporary directory after test
        shutil.rmtree(self.temp_dir)

        # Restore the original environment variables
        if self.original_tfts_home is None:
            os.environ.pop("TFTS_HOME", None)
        else:
            os.environ["TFTS_HOME"] = self.original_tfts_home

        if self.original_tfts_hub_cache is None:
            os.environ.pop("TFTS_HUB_CACHE", None)
        else:
            os.environ["TFTS_HUB_CACHE"] = self.original_tfts_hub_cache

        if self.original_tfts_assets_cache is None:
            os.environ.pop("TFTS_ASSETS_CACHE", None)
        else:
            os.environ["TFTS_ASSETS_CACHE"] = self.original_tfts_assets_cache

        if self.original_xdg_cache_home is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = self.original_xdg_cache_home

    def test_cache_paths_with_tfts_home_set(self):
        # When TFTS_HOME is set, check that it uses the correct cache path
        os.environ["TFTS_HOME"] = self.temp_dir

        TFTS_HOME = os.getenv("TFTS_HOME")
        new_default_cache_path = os.path.join(TFTS_HOME, "hub")
        new_default_assets_cache_path = os.path.join(TFTS_HOME, "assets")
        TFTS_HUB_CACHE = os.getenv("TFTS_HUB_CACHE", new_default_cache_path)
        TFTS_ASSETS_CACHE = os.getenv("TFTS_ASSETS_CACHE", new_default_assets_cache_path)

        expected_tfts_hub_cache = os.path.join(TFTS_HOME, "hub")
        expected_tfts_assets_cache = os.path.join(TFTS_HOME, "assets")

        self.assertEqual(TFTS_HUB_CACHE, expected_tfts_hub_cache)
        self.assertEqual(TFTS_ASSETS_CACHE, expected_tfts_assets_cache)
