import json
import typing as tp
from collections import OrderedDict
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def read_json(file_path: str | Path) -> OrderedDict:
    """
    Read the given json file.

    Args:
        file_path (str | Path): filename of the json file.
    Returns:
        json (list[OrderedDict] | OrderedDict): loaded json.
    """
    file_path = Path(file_path)
    with file_path.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: tp.Any, file_path: str | Path) -> None:
    """
    Write the content to the given json file.

    Args:
        content (Any JSON-friendly): content to write.
        file_path (str): filename of the json file.
    """
    file_path = Path(file_path)
    with file_path.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
