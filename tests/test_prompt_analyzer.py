import sys
import os

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from core.prompt_analyzer import analyze_prompt, CreativePromptEnhancer


def test_analyze_anime_portrait():
    res = analyze_prompt("beautiful anime portrait of a hero")
    assert res["style"] == "anime"
    assert res["orientation"] == "portrait"
    assert res["width"] == 768
    assert res["height"] == 1024


def test_analyze_realistic_landscape():
    res = analyze_prompt("realistic landscape photo of mountains")
    assert res["style"] == "realistic"
    assert res["orientation"] == "landscape"
    assert res["width"] == 1024
    assert res["height"] == 768


def test_analyze_general_square():
    res = analyze_prompt("abstract shapes")
    assert res["orientation"] == "square"


def test_creative_mode_application():
    enhancer = CreativePromptEnhancer()
    result = enhancer.apply_mode("castle", "🎮 Game Art")
    assert result["prompt"].startswith("video game concept art castle")
    assert "digital painting" in result["prompt"]
    assert result["steps"] == 30


def test_surprise_prompt_generation():
    enhancer = CreativePromptEnhancer()
    prompt = enhancer.surprise_prompt()
    assert isinstance(prompt, str) and len(prompt) > 0

