"""
Root shim for hf_utils, forwarding to kokoro.hf_utils.
"""

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
    download_model_file,
    load_custom_model,
)

__all__ = [
    "find_repo_root",
    "get_checkpoints_dir",
    "set_checkpoints_dir",
    "reset_checkpoints_dir",
    "ensure_checkpoints_dir",
    "parse_repo_id",
    "get_repo_name",
    "resolve_model_path",
    "resolve_pretrained_path",
    "download_model_file",
    "load_custom_model",
]
