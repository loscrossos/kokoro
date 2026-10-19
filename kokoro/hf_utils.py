"""
Centralized Hugging Face and Checkpoint Resolution Utilities for Kokoro TTS.

Provides unified checkpoint directory management, path resolution, offline loading,
and model downloading helpers across the entire repository.
"""

import json
import os
import shutil
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("kokoro.hf_utils")

_CHECKPOINTS_DIR: Optional[Path] = None


def find_repo_root() -> Path:
    """Finds repository root by searching upward for markers like pyproject.toml or .git."""
    current = Path(__file__).resolve().parent
    for p in [current, *current.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return Path.cwd()


def get_checkpoints_dir() -> Path:
    """
    Returns the active checkpoints directory.
    Priority:
    1. Explicitly set via set_checkpoints_dir()
    2. CHECKPOINT_DIR environment variable
    3. <repo_root>/checkpoints (if repo root found)
    4. Current working directory / checkpoints
    """
    global _CHECKPOINTS_DIR
    if _CHECKPOINTS_DIR is not None:
        return _CHECKPOINTS_DIR
    env_dir = os.environ.get("CHECKPOINT_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    repo_root = find_repo_root()
    default_dir = repo_root / "checkpoints"
    return default_dir.resolve()


def set_checkpoints_dir(path: Union[str, Path]) -> Path:
    """
    Sets the active checkpoints directory and synchronizes environment variables:
    - CHECKPOINT_DIR
    - HF_HUB_CACHE
    - HF_HOME
    """
    global _CHECKPOINTS_DIR
    p = Path(path).resolve()
    _CHECKPOINTS_DIR = p
    os.environ["CHECKPOINT_DIR"] = str(p)
    os.environ["HF_HUB_CACHE"] = str(p / "hub")
    os.environ["HF_HOME"] = str(p / "hf_home")
    return p


def reset_checkpoints_dir() -> None:
    """Resets the global checkpoint directory state (useful for tests)."""
    global _CHECKPOINTS_DIR
    _CHECKPOINTS_DIR = None


def ensure_checkpoints_dir(path: Optional[Union[str, Path]] = None) -> Path:
    """Ensures the checkpoints directory exists on disk and returns its Path."""
    p = Path(path).resolve() if path else get_checkpoints_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_repo_id(repo_id_or_url: str) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Parses a repository ID, Hugging Face URL, ModelScope URL, or Git URL into:
    (author, repo_name, subpath_or_filename)
    
    Examples:
    - 'hexgrad/Kokoro-82M' -> ('hexgrad', 'Kokoro-82M', None)
    - 'Kokoro-82M' -> (None, 'Kokoro-82M', None)
    - 'https://huggingface.co/hexgrad/Kokoro-82M' -> ('hexgrad', 'Kokoro-82M', None)
    - 'https://huggingface.co/hexgrad/Kokoro-82M.git' -> ('hexgrad', 'Kokoro-82M', None)
    - 'https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth' -> ('hexgrad', 'Kokoro-82M', 'kokoro-v1_0.pth')
    - 'https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt' -> ('hexgrad', 'Kokoro-82M', 'voices/af_heart.pt')
    - 'https://huggingface.co/author/reponame/subdir/model.ext' -> ('author', 'reponame', 'subdir/model.ext')
    - 'https://modelscope.cn/models/author/reponame' -> ('author', 'reponame', None)
    - 'https://modelscope.cn/models/author/reponame/resolve/master/weights.bin' -> ('author', 'reponame', 'weights.bin')
    """
    if not repo_id_or_url:
        return None, "", None
        
    s = str(repo_id_or_url).strip()
    
    if s.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(s)
        path = parsed.path.strip("/")
        parts = [p for p in path.split("/") if p]
        domain = parsed.netloc.lower()
        
        if "huggingface.co" in domain or "hf.co" in domain:
            if len(parts) >= 2:
                author = parts[0]
                repo_name = parts[1].removesuffix(".git")
                subpath = None
                if len(parts) >= 5 and parts[2] in ("resolve", "raw", "blob"):
                    subpath = "/".join(parts[4:])
                elif len(parts) > 2:
                    subpath = "/".join(parts[2:])
                return author, repo_name, subpath
            elif len(parts) == 1:
                return None, parts[0].removesuffix(".git"), None
                
        elif "modelscope.cn" in domain:
            if len(parts) >= 3 and parts[0] == "models":
                author = parts[1]
                repo_name = parts[2].removesuffix(".git")
                subpath = None
                if len(parts) >= 6 and parts[3] in ("resolve", "raw", "blob"):
                    subpath = "/".join(parts[5:])
                elif len(parts) > 3:
                    subpath = "/".join(parts[3:])
                return author, repo_name, subpath
            elif len(parts) >= 2:
                author = parts[0]
                repo_name = parts[1].removesuffix(".git")
                subpath = "/".join(parts[2:]) if len(parts) > 2 else None
                return author, repo_name, subpath
        else:
            if len(parts) >= 2:
                author = parts[0]
                repo_name = parts[1].removesuffix(".git")
                subpath = None
                if len(parts) >= 5 and parts[2] in ("releases", "raw", "blob") and parts[3] in ("download", "main", "master"):
                    subpath = "/".join(parts[4:])
                elif len(parts) > 2:
                    subpath = "/".join(parts[2:])
                return author, repo_name, subpath
            elif len(parts) == 1:
                return None, parts[0].removesuffix(".git"), None

    s = s.removesuffix(".git")
    parts = [p for p in s.split("/") if p]
    if len(parts) == 1:
        return None, parts[0], None
    elif len(parts) == 2:
        return parts[0], parts[1], None
    else:
        return parts[0], parts[1], "/".join(parts[2:])


def get_repo_name(repo_id_or_url: str) -> str:
    """Extracts the repository name from a repo ID or URL."""
    _, repo_name, _ = parse_repo_id(repo_id_or_url)
    return repo_name


def resolve_model_path(
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    base_dir: Optional[Union[str, Path]] = None,
    must_exist: bool = False
) -> Optional[Path]:
    """
    Checks if a local checkpoint file exists at <base_dir>/<reponame>/<subpath>/<filename>
    (plus any fallback paths).
    
    Fallback paths checked:
    1. Direct file path if filename or repo_id is an existing file
    2. Primary: <base_dir>/<reponame>/<subpath>/<filename>
    3. Fallback: <base_dir>/<author>/<reponame>/<filename>
    4. Fallback: Flat in <base_dir>/<filename>
    5. Fallback: In current working directory or relative path
    """
    base = Path(base_dir).resolve() if base_dir else get_checkpoints_dir()
    
    author, repo_name, parsed_subpath = parse_repo_id(repo_id) if repo_id else (None, "", None)
    
    target_file = filename
    if target_file is None and parsed_subpath:
        target_file = parsed_subpath
        
    candidates = []
    
    # 1. Direct path check
    if target_file:
        p = Path(target_file)
        if p.is_file():
            return p.resolve()
        p_base = base / target_file
        if p_base.is_file():
            candidates.append(p_base)
            
    # 2. Primary candidate: <base_dir>/<repo_name>/<target_file>
    if repo_name and target_file:
        candidates.append(base / repo_name / target_file)
    elif repo_name and not target_file:
        candidates.append(base / repo_name)
    elif target_file:
        candidates.append(base / target_file)
        
    # 3. Fallback: <base_dir>/<author>/<repo_name>/<target_file>
    if author and repo_name and target_file:
        candidates.append(base / author / repo_name / target_file)
        
    # 4. Fallback: Flat in base_dir
    if target_file:
        flat = base / Path(target_file).name
        if flat not in candidates:
            candidates.append(flat)
            
    # 5. Fallback: relative to cwd
    if repo_name and target_file:
        candidates.append(Path.cwd() / repo_name / target_file)
    if target_file:
        candidates.append(Path.cwd() / target_file)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
            
    if must_exist:
        return None
        
    if repo_name and target_file:
        return base / repo_name / target_file
    elif target_file:
        return base / target_file
    elif repo_name:
        return base / repo_name
    return base


def resolve_pretrained_path(
    model_id: str,
    base_dir: Optional[Union[str, Path]] = None,
    must_exist: bool = False
) -> Union[Path, str]:
    """
    Resolves pretrained directories for .from_pretrained() calls
    (e.g., Hugging Face transformers, BigVGAN, vocoders).
    """
    if not model_id:
        return model_id
        
    p = Path(model_id)
    if p.is_dir():
        return p.resolve()
        
    base = Path(base_dir).resolve() if base_dir else get_checkpoints_dir()
    author, repo_name, _ = parse_repo_id(model_id)
    
    candidates = []
    if repo_name:
        candidates.append(base / repo_name)
    if author and repo_name:
        candidates.append(base / author / repo_name)
    
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
            
    if (base / "config.json").is_file():
        return base
            
    if must_exist:
        return model_id
        
    return (base / repo_name) if repo_name else base


def _load_file(path: Union[str, Path], loader: Optional[Union[str, Callable]] = None, **kwargs) -> Any:
    """Helper to load a resolved file using an appropriate loader."""
    p = Path(path)
    if loader == "path":
        return str(p)
    if loader == "Path":
        return p
    if callable(loader):
        return loader(str(p), **kwargs)
        
    suffix = p.suffix.lower()
    if suffix in (".pth", ".pt"):
        import torch
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = True
        if "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return torch.load(str(p), **kwargs)
    elif suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    elif suffix == ".bin":
        with open(p, "rb") as f:
            return f.read()
    return str(p)


def download_model_file(
    repo_id: str,
    filename: str,
    base_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False
) -> Path:
    """Downloads a model file directly into <base_dir>/<reponame>/<filename>."""
    base = Path(base_dir).resolve() if base_dir else get_checkpoints_dir()
    repo_name = get_repo_name(repo_id)
    target_dir = base / repo_name
    target_file = target_dir / filename
    
    if not force_download and target_file.is_file():
        logger.debug(f"File already exists locally at {target_file}")
        return target_file
        
    target_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {filename} from {repo_id} to {target_file}")
    
    from huggingface_hub import hf_hub_download
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.is_file():
            return downloaded_path
    except TypeError:
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copyfile(downloaded, target_file)
        return target_file
        
    return target_file


def load_custom_model(
    repo_id: str,
    filename: Optional[str] = None,
    base_dir: Optional[Union[str, Path]] = None,
    loader: Optional[Union[str, Callable]] = None,
    force_download: bool = False,
    **loader_kwargs
) -> Any:
    """
    If the model exists locally, loads it immediately without network calls;
    if absent, downloads it into the designated <base_dir>/<reponame>/ destination.
    """
    base = Path(base_dir).resolve() if base_dir else get_checkpoints_dir()
    
    if filename is None:
        if "Kokoro-82M-v1.1-zh" in repo_id:
            filename = "kokoro-v1_1-zh.pth"
        elif "Kokoro-82M" in repo_id:
            filename = "kokoro-v1_0.pth"
        else:
            filename = "config.json"
            
    local_path = None if force_download else resolve_model_path(repo_id, filename, base_dir=base, must_exist=True)
    
    if local_path is not None and local_path.is_file():
        logger.info(f"Loading local checkpoint from {local_path}")
        return _load_file(local_path, loader=loader, **loader_kwargs)
        
    repo_name = get_repo_name(repo_id)
    target_dir = base / repo_name
    target_file = target_dir / filename
    target_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Checkpoint not found locally. Downloading {filename} from {repo_id} to {target_file}")
    
    try:
        from huggingface_hub import hf_hub_download
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.is_file():
                local_path = downloaded_path
            elif target_file.is_file():
                local_path = target_file
            else:
                local_path = downloaded_path
        except TypeError:
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copyfile(downloaded, target_file)
            local_path = target_file
    except Exception as e:
        if target_file.is_file():
            local_path = target_file
        else:
            raise RuntimeError(f"Failed to download {filename} from {repo_id} to {target_file}: {e}") from e
            
    return _load_file(local_path, loader=loader, **loader_kwargs)
