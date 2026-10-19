import csv
from pathlib import Path
from typing import Callable, Optional


class EpochLogger:
    def __init__(
        self,
        log_file_path: str,
        model_name: str,
        commit_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self.log_file_path = log_file_path
        self.model_name = model_name
        self.commit_fn = commit_fn
        self.logs_dict: dict[int, dict[str, object]] = {}

    def save_csv(self) -> None:
        path = Path(self.log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [self.logs_dict[key] for key in sorted(self.logs_dict)]
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        if not fieldnames:
            fieldnames = ["Epoch"]

        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if self.commit_fn is not None:
            self.commit_fn()
