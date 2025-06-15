import os
import sys
import json
import zipfile
from pathlib import Path
from PIL import Image

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def test_export_gallery_contains_images_and_metadata(tmp_path, monkeypatch):
    import importlib, importlib.util, types
    for mod in ['core.sdxl', 'core.state', 'core.memory', 'core.memory_guardian', 'core']:
        sys.modules.pop(mod, None)
    monkeypatch.syspath_prepend(os.getcwd())
    pkg = types.ModuleType('core')
    pkg.__path__ = [os.path.join(os.getcwd(), 'core')]
    sys.modules['core'] = pkg
    sys.modules['core.image_generator'] = types.ModuleType('core.image_generator')
    sys.modules['core.image_generator'].ImageGenerator = object
    spec = importlib.util.spec_from_file_location('core.sdxl', os.path.join('core', 'sdxl.py'))
    sdxl = importlib.util.module_from_spec(spec)
    sdxl.SDXLConfig = type('SDXLConfig', (), {})
    sys.modules['core.sdxl'] = sdxl
    spec.loader.exec_module(sdxl)
    state_mod = importlib.import_module('core.state')
    AppState = state_mod.AppState

    gallery_dir = tmp_path / "gallery"
    monkeypatch.setattr(sdxl, "GALLERY_DIR", gallery_dir)

    state = AppState()
    img1 = Image.new("RGB", (10, 10), color="red")
    img2 = Image.new("RGB", (10, 10), color="blue")
    path1 = Path(sdxl.save_to_gallery(state, img1, "prompt1"))
    path2 = Path(sdxl.save_to_gallery(state, img2, "prompt2"))

    zip_path = Path(sdxl.export_gallery(state))
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for img_path in [path1, path2]:
            assert img_path.name in names
            meta_name = img_path.with_suffix(".json").name
            assert meta_name in names
            with zf.open(meta_name) as f:
                data = json.load(f)
                assert "prompt" in data
