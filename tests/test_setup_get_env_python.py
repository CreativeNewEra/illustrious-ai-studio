import types
import platform
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "setup_script", Path(__file__).resolve().parents[1] / "src/illustrious_ai_studio/setup_cli.py"
)
setup_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_script)
IllustriousSetup = setup_script.IllustriousSetup


def test_get_env_python_returns_path(monkeypatch):
    args = types.SimpleNamespace(verbose=False)
    setup = IllustriousSetup(args)
    setup.conda_exe = '/usr/bin/conda'

    def fake_run_command(cmd, **kwargs):
        return True, '/tmp/env/bin/python\n'

    monkeypatch.setattr(setup, 'run_command', fake_run_command)
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')

    python_path = setup.get_env_python()
    assert python_path == '/tmp/env/bin/python'
