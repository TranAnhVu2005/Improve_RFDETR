"""Download the configured Roboflow dataset in COCO format."""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="nckhcict2025")
    parser.add_argument("--project", default="completed-project")
    parser.add_argument("--version", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets") / "completed-project-5" / "coco",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set ROBOFLOW_API_KEY in the environment before downloading."
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(
        "coco", location=str(args.output_dir)
    )
    print(f"Dataset downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()
