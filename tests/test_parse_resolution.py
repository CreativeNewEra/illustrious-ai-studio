import os
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import importlib
import types
from pathlib import Path


def stub_core_modules():
    core_pkg = types.ModuleType('core')
    sys.modules.setdefault('core', core_pkg)

    sdxl = types.ModuleType('core.sdxl')
    sdxl.generate_image = lambda *a, **k: None
    sdxl.generate_with_notifications = lambda *a, **k: None
    sdxl.TEMP_DIR = Path('/tmp')
    sdxl.get_latest_image = lambda *a, **k: None
    sdxl.init_sdxl = lambda *a, **k: None
    sdxl.get_available_models = lambda *a, **k: []
    sdxl.get_current_model_info = lambda *a, **k: {}
    sdxl.test_model_generation = lambda *a, **k: None
    sdxl.switch_sdxl_model = lambda *a, **k: None
    sdxl.PROJECTS_DIR = Path('/tmp')
    sys.modules['core.sdxl'] = sdxl
    setattr(core_pkg, 'sdxl', sdxl)

    config = types.ModuleType('core.config')
    config.CONFIG = {}
    sys.modules['core.config'] = config
    setattr(core_pkg, 'config', config)

    state = types.ModuleType('core.state')
    class DummyState: ...
    state.AppState = DummyState
    sys.modules['core.state'] = state
    setattr(core_pkg, 'state', state)

    ollama = types.ModuleType('core.ollama')
    ollama.generate_prompt = lambda *a, **k: None
    ollama.handle_chat = lambda *a, **k: None
    ollama.analyze_image = lambda *a, **k: None
    ollama.init_ollama = lambda *a, **k: None
    sys.modules['core.ollama'] = ollama
    setattr(core_pkg, 'ollama', ollama)

    gp = types.ModuleType('core.generation_presets')
    gp.GENERATION_PRESETS = {}
    gp.DEFAULT_PRESET = {}
    sys.modules['core.generation_presets'] = gp
    setattr(core_pkg, 'generation_presets', gp)

    memory = types.ModuleType('core.memory')
    memory.get_model_status = lambda *a, **k: None
    memory.get_memory_stats_markdown = lambda *a, **k: None
    memory.get_memory_stats_wrapper = lambda *a, **k: None
    sys.modules['core.memory'] = memory
    setattr(core_pkg, 'memory', memory)

    mg = types.ModuleType('core.memory_guardian')
    mg.start_memory_guardian = lambda *a, **k: None
    mg.stop_memory_guardian = lambda *a, **k: None
    mg.get_memory_guardian = lambda *a, **k: None
    sys.modules['core.memory_guardian'] = mg
    setattr(core_pkg, 'memory_guardian', mg)

    pt = types.ModuleType('core.prompt_templates')
    pt.template_manager = None
    sys.modules['core.prompt_templates'] = pt
    setattr(core_pkg, 'prompt_templates', pt)

    pa = types.ModuleType('core.prompt_analyzer')
    pa.analyze_prompt = lambda *a, **k: None
    sys.modules['core.prompt_analyzer'] = pa
    setattr(core_pkg, 'prompt_analyzer', pa)

    gf = types.ModuleType('core.gallery_filters')
    gf.load_gallery_filter = lambda *a, **k: None
    gf.save_gallery_filter = lambda *a, **k: None
    sys.modules['core.gallery_filters'] = gf
    setattr(core_pkg, 'gallery_filters', gf)


def load_web_module():
    stub_core_modules()
    if 'ui.web' in sys.modules:
        return importlib.reload(sys.modules['ui.web'])
    return importlib.import_module('ui.web')


def test_parse_resolution_empty_string():
    web = load_web_module()
    assert web.parse_resolution("") == (1024, 1024)
