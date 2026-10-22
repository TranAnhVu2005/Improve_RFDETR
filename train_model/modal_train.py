"""Modal entrypoint for baseline or improved RF-DETR training."""

import importlib
import os
import shutil
import tempfile
from pathlib import Path

import modal


APP_NAME = "rfdetr-baseline"
VOLUME_NAME = os.environ.get("RFDETR_VOLUME_NAME", "rfdetr-storage")
ROBOFLOW_SECRET_NAME = os.environ.get(
    "ROBOFLOW_SECRET_NAME", "roboflow-api-key")
REMOTE_ROOT = "/workspace"
VOLUME_ROOT = "/vol"
DEFAULT_DATASET_DIR = f"{VOLUME_ROOT}/datasets/completed-project-5/coco"
ROBOFLOW_WORKSPACE = "nckhcict2025"
ROBOFLOW_PROJECT = "completed-project"
ROBOFLOW_VERSION = 5

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
roboflow_secret = modal.Secret.from_name(ROBOFLOW_SECRET_NAME)


def _build_dataset_image() -> modal.Image:
    return modal.Image.debian_slim(python_version="3.11").pip_install("roboflow")


def _build_image(rfdetr_requirement: str) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(rfdetr_requirement, "PyYAML", "pandas", "tensorboard")
        .workdir(REMOTE_ROOT)
        .add_local_dir(
            str(Path(__file__).resolve().parents[1]),
            remote_path=REMOTE_ROOT,
            ignore=[
                ".git",
                ".venv",
                ".env",
                ".env.*",
                "**/.env",
                "**/.env.*",
                "datasets/**",
                "checkpoints/**",
                "logs/**",
                "outputs/**",
                "**/__pycache__",
                "**/*.pth",
                "**/events.out.tfevents.*",
            ],
        )
    )


image = _build_image("rfdetr[train,loggers]")
improved_image = _build_image("rfdetr[train,loggers]==1.11.2")
dataset_image = _build_dataset_image()


def _is_valid_coco_dataset(path: Path) -> bool:
    return all(
        (path / split / "_annotations.coco.json").is_file()
        for split in ("train", "valid")
    )


def _find_coco_dataset(root: Path) -> Path | None:
    if _is_valid_coco_dataset(root):
        return root
    for candidate in root.rglob("*"):
        if candidate.is_dir() and _is_valid_coco_dataset(candidate):
            return candidate
    return None


@app.function(
    image=dataset_image,
    timeout=60 * 60,
    volumes={VOLUME_ROOT: volume},
    secrets=[roboflow_secret],
)
def ensure_roboflow_dataset(dataset_dir: str = DEFAULT_DATASET_DIR) -> str:
    destination = Path(dataset_dir)
    if _is_valid_coco_dataset(destination):
        print(f"Using existing Roboflow COCO dataset: {destination}")
        return str(destination)

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Modal Secret '{ROBOFLOW_SECRET_NAME}' must contain ROBOFLOW_API_KEY."
        )

    Roboflow = importlib.import_module("roboflow").Roboflow

    print(
        f"Dataset not found at {destination}; downloading "
        f"{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION} in COCO format."
    )
    with tempfile.TemporaryDirectory(prefix="roboflow-coco-") as temp_dir:
        # Roboflow skips downloading when the requested location already exists.
        # Use a new child path because TemporaryDirectory itself already exists.
        download_location = Path(temp_dir) / "dataset"
        downloaded = (
            Roboflow(api_key=api_key)
            .workspace(ROBOFLOW_WORKSPACE)
            .project(ROBOFLOW_PROJECT)
            .version(ROBOFLOW_VERSION)
            .download("coco", location=str(download_location))
        )
        download_root = Path(
            getattr(downloaded, "location", download_location))
        source = _find_coco_dataset(download_root)
        if source is None:
            raise FileNotFoundError(
                "Roboflow download completed, but train/valid COCO annotations "
                f"were not found under {download_root}."
            )

        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)

    if not _is_valid_coco_dataset(destination):
        raise FileNotFoundError(
            "Downloaded dataset is missing train/valid COCO annotations "
            f"under {destination}."
        )

    volume.commit()
    print(f"Dataset saved to Modal Volume: {destination}")
    return str(destination)


@app.function(
    image=image,
    gpu="A100",
    timeout=24 * 60 * 60,
    volumes={VOLUME_ROOT: volume},
)
def train_baseline(
    dataset_dir: str = "/vol/datasets/completed-project-5/coco",
    config_path: str = "/workspace/config/rfdetr_baseline.yaml",
) -> str:
    from train_model.train_baseline import train_rfdetr

    return train_rfdetr(
        dataset_dir=dataset_dir,
        config_path=config_path,
        commit_fn=volume.commit,
    )


@app.function(
    image=improved_image,
    gpu="A100",
    timeout=24 * 60 * 60,
    volumes={VOLUME_ROOT: volume},
)
def train_improved(
    dataset_dir: str = "/vol/datasets/completed-project-5/coco",
    config_path: str = "/workspace/config/rfdetr_improve_v1.yaml",
) -> str:
    from train_model.train_improve_v1 import train_rfdetr

    return train_rfdetr(
        dataset_dir=dataset_dir,
        config_path=config_path,
        commit_fn=volume.commit,
    )


@app.local_entrypoint()
def main(
    mode: str = "baseline",
    dataset_dir: str = DEFAULT_DATASET_DIR,
    config_path: str | None = None,
) -> None:
    if mode not in ("baseline", "improve"):
        raise ValueError("mode must be 'baseline' or 'improve'.")

    dataset_dir = ensure_roboflow_dataset.remote(dataset_dir)
    if mode == "baseline":
        checkpoint = train_baseline.remote(
            dataset_dir=dataset_dir,
            config_path=config_path or "/workspace/config/rfdetr_baseline.yaml",
        )
    elif mode == "improve":
        checkpoint = train_improved.remote(
            dataset_dir=dataset_dir,
            config_path=config_path or "/workspace/config/rfdetr_improve_v1.yaml",
        )
    print(f"Training finished. Best checkpoint: {checkpoint}")
