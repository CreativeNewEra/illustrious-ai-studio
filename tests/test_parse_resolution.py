import sys

import importlib
from pathlib import Path

from conftest import stub_core_modules


def load_web_module():
    stub_core_modules()
    if 'ui.web' in sys.modules:
        return importlib.reload(sys.modules['ui.web'])
    return importlib.import_module('ui.web')


def test_parse_resolution_empty_string():
    web = load_web_module()
    assert web.parse_resolution("") == (1024, 1024)
