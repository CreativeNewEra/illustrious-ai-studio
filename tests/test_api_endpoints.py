import base64
import io
from PIL import Image
from fastapi.testclient import TestClient
import types
import importlib
import sys
import pytest

def load_app():
    
    if 'gradio' not in sys.modules:
        sys.modules['gradio'] = types.ModuleType('gradio')
    if 'diffusers' not in sys.modules:
        diff = types.ModuleType('diffusers')
        diff.StableDiffusionXLPipeline = object
        sys.modules['diffusers'] = diff
    if 'torch' not in sys.modules:
        torch = types.ModuleType('torch')
        class DummyCuda:
            @staticmethod
            def is_available():
                return False
            @staticmethod
            def empty_cache():
                pass
            @staticmethod
            def synchronize():
                pass
        torch.cuda = DummyCuda()
        class DummyGenerator:
            def __init__(self, device=None):
                pass
            def manual_seed(self, seed):
                pass
            def initial_seed(self):
                return 0
        torch.Generator = DummyGenerator
        torch.float16 = 'float16'
        sys.modules['torch'] = torch
    return importlib.import_module('illustrious_ai_studio.app')

# Dummy implementations
class DummyPipe:
    def generate(self, *args, **kwargs):
        return types.SimpleNamespace(images=[Image.new('RGB', (64, 64), 'blue')])

async def dummy_chat_completion(messages, temperature=0.7, max_tokens=256):
    return "ok"

async def dummy_analyze_image(image, question=""):
    return "blue"

@pytest.fixture(autouse=True)
def setup_app(monkeypatch):
    app = load_app()
    app.app_state.sdxl_pipe = DummyPipe()
    app.app_state.ollama_model = 'dummy'
    app.app_state.model_status.update({'sdxl': True, 'ollama': True, 'multimodal': True})
    import illustrious_ai_studio.server.api as api
    import illustrious_ai_studio.server.tasks as tasks
    monkeypatch.setattr(api, 'generate_image', lambda state, params: (Image.new('RGB',(64,64),'blue'), 'done'))
    monkeypatch.setattr(api, 'chat_completion', lambda state, *a, **k: dummy_chat_completion(*a, **k))
    monkeypatch.setattr(api, 'analyze_image', lambda state, img, q='': dummy_analyze_image(img, q))
    monkeypatch.setattr(app, 'clear_gpu_memory', lambda: None)
    monkeypatch.setattr(sys.modules['__main__'], 'app', app, raising=False)

    results = {}

    class DummyResult:
        def __init__(self, data):
            self.id = 'job1'
            self.data = data
            self.state = 'SUCCESS'

        @property
        def result(self):
            return self.data

    def fake_delay(params):
        img = Image.new('RGB', (64, 64), 'blue')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode()
        result = {'success': True, 'image_base64': encoded, 'message': 'done'}
        results['job1'] = result
        return DummyResult(result)

    monkeypatch.setattr(tasks.generate_image_task, 'delay', fake_delay)
    monkeypatch.setattr(api, 'AsyncResult', lambda job_id, app=None: types.SimpleNamespace(state='SUCCESS', result=results[job_id]))
    yield

def get_client():
    return TestClient(load_app().app)


def test_status_endpoint():
    client = get_client()
    response = client.get('/status')
    assert response.status_code == 200
    assert response.json()['status'] == 'running'


def test_generate_image_endpoint():
    client = get_client()
    data = {"prompt": "hi"}
    resp = client.post('/generate-image', json=data)
    assert resp.status_code == 200
    job = resp.json()
    assert 'job_id' in job
    status = client.get(f"/status/{job['job_id']}")
    assert status.status_code == 200
    data = status.json()
    assert data['state'] == 'SUCCESS'
    assert 'image_base64' in data['result']


def test_chat_endpoint():
    client = get_client()
    data = {"message": "hello"}
    resp = client.post('/chat', json=data)
    assert resp.status_code == 200
    assert resp.json()['response'] == 'ok'


def test_analyze_image_endpoint():
    client = get_client()
    # create dummy image
    img = Image.new('RGB', (10,10), 'blue')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode()
    data = {"image_base64": encoded, "question": "what"}
    resp = client.post('/analyze-image', json=data)
    assert resp.status_code == 200
    assert resp.json()['analysis'] == 'blue'


def test_concurrent_requests():
    client = get_client()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def send(i):
        resp = client.post('/chat', json={"message": f"hi {i}"})
        return resp.status_code
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(send, i) for i in range(10)]
        results = [f.result() for f in as_completed(futures)]
    assert all(r == 200 for r in results)

def test_generate_image_endpoint_no_model():
    app = load_app()
    app.app_state.sdxl_pipe = None
    client = TestClient(app.app)
    resp = client.post('/generate-image', json={"prompt": "x"})
    assert resp.status_code == 503


def test_chat_endpoint_no_model():
    app = load_app()
    app.app_state.ollama_model = None
    client = TestClient(app.app)
    resp = client.post('/chat', json={"message": "hi"})
    assert resp.status_code == 503


def test_analyze_image_invalid_input():
    app = load_app()
    client = TestClient(app.app)
    app.app_state.ollama_model = 'dummy'
    app.app_state.model_status['multimodal'] = True
    resp = client.post('/analyze-image', json={"image_base64": "notbase64"})
    assert resp.status_code == 400


def test_switch_models_endpoint(monkeypatch):
    client = get_client()
    calls = {}
    import illustrious_ai_studio.server.api as api
    monkeypatch.setattr(api.sdxl, 'switch_sdxl_model', lambda state, p: calls.setdefault('sdxl', p) or True)
    monkeypatch.setattr(api.ollama, 'switch_ollama_model', lambda state, n: calls.setdefault('ollama', n) or True)
    resp = client.post('/switch-models', json={"sd_model": "a", "ollama_model": "b"})
    assert resp.status_code == 200
    assert calls['sdxl'] == 'a'
    assert calls['ollama'] == 'b'


def test_memory_profile_endpoint():
    client = get_client()
    resp = client.post('/memory-profile', json={"profile": "conservative"})
    assert resp.status_code == 200
    assert resp.json()["profile"] == "conservative"


def test_memory_thresholds_endpoint():
    client = get_client()
    resp = client.post('/memory-thresholds', json={"low": 60, "critical": 99})
    assert resp.status_code == 200
    data = resp.json()
    assert data["low"] == "60%"
    assert data["critical"] == "99%"
