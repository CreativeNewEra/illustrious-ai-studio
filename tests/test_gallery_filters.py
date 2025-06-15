import json
import os
from pathlib import Path

# Removed global modification of sys.path



def test_save_and_load_gallery_filter(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(os.getcwd())
    from core import gallery_filters

    filter_file = tmp_path / "filter.json"
    monkeypatch.setattr(gallery_filters, "GALLERY_FILTER_FILE", filter_file)

    data = {"tag": "portrait", "rating": 5}
    gallery_filters.save_gallery_filter(data)

    assert json.loads(filter_file.read_text()) == data

    loaded = gallery_filters.load_gallery_filter()
    assert loaded == data

