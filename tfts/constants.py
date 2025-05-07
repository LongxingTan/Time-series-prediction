import os

# default cache
default_home = os.path.join(os.path.expanduser("~"), ".cache")
TFTS_HOME = os.path.expanduser(
    os.getenv(
        "TFTS_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "tfts"),
    )
)

# model will be saved in TFTS_HOME/hub, and assets will be saved in TFTS_HOME/assets
default_cache_path = os.path.join(TFTS_HOME, "hub")
default_assets_cache_path = os.path.join(TFTS_HOME, "assets")

TFTS_HUB_CACHE = os.getenv("TFTS_HUB_CACHE", default_cache_path)
TFTS_ASSETS_CACHE = os.getenv("TFTS_ASSETS_CACHE", default_assets_cache_path)

TF2_WEIGHTS_NAME = "tf_model.h5"
TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
TF_WEIGHTS_NAME = "model.ckpt"
CONFIG_NAME = "config.json"
