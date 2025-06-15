import os
import sys
import ast
import importlib.util
import logging
import types
from pathlib import Path

# Ensure project root is on the path for imports
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def _stub_core_modules():
    core_pkg = sys.modules.get("core")
    core_path = str(Path(__file__).resolve().parent.parent / "core")
    if core_pkg is None:
        core_pkg = types.ModuleType("core")
        sys.modules["core"] = core_pkg
    core_pkg.__path__ = [core_path]

    sdxl = types.ModuleType("core.sdxl")
    sdxl.generate_image = lambda *a, **k: None
    sdxl.generate_with_notifications = lambda *a, **k: None
    sdxl.TEMP_DIR = Path("/tmp")
    sdxl.get_latest_image = lambda *a, **k: None
    sdxl.init_sdxl = lambda *a, **k: None
    sdxl.get_available_models = lambda *a, **k: []
    sdxl.get_current_model_info = lambda *a, **k: {}
    sdxl.test_model_generation = lambda *a, **k: None
    sdxl.switch_sdxl_model = lambda *a, **k: None
    sdxl.save_to_gallery = lambda *a, **k: None
    sdxl.PROJECTS_DIR = Path("/tmp")
    sys.modules["core.sdxl"] = sdxl
    setattr(core_pkg, "sdxl", sdxl)

    config = types.ModuleType("core.config")
    config.CONFIG = {}
    sys.modules["core.config"] = config
    setattr(core_pkg, "config", config)

    state = types.ModuleType("core.state")
    class DummyState: ...
    state.AppState = DummyState
    sys.modules["core.state"] = state
    setattr(core_pkg, "state", state)

    ollama = types.ModuleType("core.ollama")
    ollama.generate_prompt = lambda *a, **k: None
    ollama.handle_chat = lambda *a, **k: None
    ollama.analyze_image = lambda *a, **k: None
    ollama.init_ollama = lambda *a, **k: None
    sys.modules["core.ollama"] = ollama
    setattr(core_pkg, "ollama", ollama)

    gp = types.ModuleType("core.generation_presets")
    gp.GENERATION_PRESETS = {}
    gp.DEFAULT_PRESET = {}
    sys.modules["core.generation_presets"] = gp
    setattr(core_pkg, "generation_presets", gp)

    memory = types.ModuleType("core.memory")
    memory.get_model_status = lambda *a, **k: None
    memory.get_memory_stats_markdown = lambda *a, **k: None
    memory.get_memory_stats_wrapper = lambda *a, **k: None
    sys.modules["core.memory"] = memory
    setattr(core_pkg, "memory", memory)

    mg = types.ModuleType("core.memory_guardian")
    mg.start_memory_guardian = lambda *a, **k: None
    mg.stop_memory_guardian = lambda *a, **k: None
    mg.get_memory_guardian = lambda *a, **k: None
    sys.modules["core.memory_guardian"] = mg
    setattr(core_pkg, "memory_guardian", mg)

    pt = types.ModuleType("core.prompt_templates")
    pt.template_manager = None
    sys.modules["core.prompt_templates"] = pt
    setattr(core_pkg, "prompt_templates", pt)

    pa = types.ModuleType("core.prompt_analyzer")
    pa.analyze_prompt = lambda *a, **k: None
    sys.modules["core.prompt_analyzer"] = pa
    setattr(core_pkg, "prompt_analyzer", pa)

    gf = types.ModuleType("core.gallery_filters")
    gf.load_gallery_filter = lambda *a, **k: None
    gf.save_gallery_filter = lambda *a, **k: None
    sys.modules["core.gallery_filters"] = gf
    setattr(core_pkg, "gallery_filters", gf)


def _parse_resolution():
    """Load the parse_resolution function from ui/web.py without importing the entire module."""
    _stub_core_modules()
    file_path = Path(__file__).resolve().parent.parent / "ui" / "web.py"
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
