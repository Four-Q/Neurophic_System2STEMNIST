"""无需完整训练即可运行的项目完整性检查。"""

from hashlib import md5, sha256
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import STEMNIST_CSNN


def file_digest(path, algorithm):
    digest = algorithm()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archives():
    pressure_zip = PROJECT_ROOT / "data" / "STEMNIST Dataset.zip"
    spike_zip = PROJECT_ROOT / "data" / "spike.zip"
    assert file_digest(pressure_zip, md5) == "6ca4638b2f95bf34f59873ab62399bd8"
    assert file_digest(spike_zip, sha256) == (
        "9f442fa3be3aafb584232f2c50b4d54359409e9f42c14339c34f334310780b0a"
    )


def validate_notebooks():
    notebook_paths = sorted(PROJECT_ROOT.rglob("*.ipynb"))
    assert len(notebook_paths) == 4
    for path in notebook_paths:
        with path.open("r", encoding="utf-8") as file:
            notebook = json.load(file)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{path}:cell-{index}", "exec")


def validate_model():
    torch.manual_seed(42)
    model = STEMNIST_CSNN(backend="torch")
    assert model.parameter_count() == 97_027
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(4, 1, 1, 16, 16))
    assert tuple(logits.shape) == (1, 35)


if __name__ == "__main__":
    validate_archives()
    validate_notebooks()
    validate_model()
    print("STEMNIST_Ready 完整性检查通过。")

