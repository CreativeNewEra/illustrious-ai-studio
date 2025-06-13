import os
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from core.prompt_templates import PromptTemplateManager


def test_nested_template_directory(tmp_path):
    nested_dir = tmp_path / "nested" / "dir"
    manager = PromptTemplateManager(templates_dir=nested_dir)
    assert manager.templates_dir == nested_dir
    assert nested_dir.exists()
