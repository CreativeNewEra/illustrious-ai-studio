import ast
import importlib.util
import logging
import types

from conftest import stub_core_modules
from pathlib import Path

# Ensure project root is on the path for imports


# The `stub_core_modules` helper is defined in `conftest.py`.
# Import and call it in test helpers or tests as needed.
def _parse_resolution():
    """Load the parse_resolution function from ui/web.py without importing the entire module."""
    stub_core_modules()
    file_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "illustrious_ai_studio"
        / "ui"
        / "web.py"
    )
    source = file_path.read_text()
    tree = ast.parse(source, filename=str(file_path))
    func_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_resolution")
    mod = ast.Module(body=[func_node], type_ignores=[])
    code = compile(mod, filename=str(file_path), mode="exec")
    namespace = {"logger": logging.getLogger(__name__)}
    exec(code, namespace)
    return namespace["parse_resolution"]


def test_parse_resolution():
    parse_resolution = _parse_resolution()
    test_cases = [
        ("1024x1024 (Square - High Quality)", (1024, 1024)),
        ("768x512 (Landscape)", (768, 512)),
        ("512x768 (Portrait)", (512, 768)),
        ("", (1024, 1024)),
        ("   ", (1024, 1024)),
        (None, (1024, 1024)),
        ("invalid", (1024, 1024)),
        ("1024", (1024, 1024)),
        ("x768", (1024, 1024)),
        ("abcxdef", (1024, 1024)),
        ("1024x", (1024, 1024)),
        ("x1024", (1024, 1024)),
        ("1024x768x512", (1024, 1024)),
        ("0x0", (1024, 1024)),
        ("-100x200", (1024, 1024)),
        ("100x-200", (1024, 1024)),
        ("9999x9999", (2048, 2048)),
        ("50x50", (256, 256)),
        ("1024.5x768.7", (1024, 1024)),
    ]

    for inp, expected in test_cases:
        assert parse_resolution(inp) == expected


def test_memory_issue_scenario():
    parse_resolution = _parse_resolution()
    problematic_inputs = ["", None, "malformed resolution string", "   \t\n   "]

    for inp in problematic_inputs:
        assert parse_resolution(inp) == (1024, 1024)
