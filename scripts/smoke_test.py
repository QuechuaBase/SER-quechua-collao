from __future__ import annotations

import ast
from pathlib import Path


def parse_python_files(root: Path) -> None:
    for directory in (root / "src", root / "scripts"):
        for path in sorted(directory.glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dictionary = root / "resources" / "qxp_v2" / "dict.ltr.txt"
    if not dictionary.is_file():
        raise FileNotFoundError(f"Missing Fairseq dictionary: {dictionary}")

    parse_python_files(root)
    print("Smoke test passed: Python files parse and small resources are available.")


if __name__ == "__main__":
    main()
