import os
from pathlib import Path
from PIL import Image



def test_save_to_gallery_creates_directory(tmp_path, monkeypatch):
    from illustrious_ai_studio.core import sdxl
    from illustrious_ai_studio.core.state import AppState

    gallery_dir = tmp_path / "gallery"
    # Ensure directory does not exist beforehand
    assert not gallery_dir.exists()

    monkeypatch.setattr(sdxl, "GALLERY_DIR", gallery_dir)

    state = AppState()
    img = Image.new("RGB", (10, 10), color="red")
    saved_path = sdxl.save_to_gallery(state, img, "test prompt")

    assert gallery_dir.exists()
    assert os.path.exists(saved_path)
    metadata_file = Path(saved_path).with_suffix('.json')
    assert metadata_file.exists()
