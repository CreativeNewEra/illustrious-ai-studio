from illustrious_ai_studio.core.prompt_analyzer import analyze_prompt


def test_analyze_anime_portrait():
    res = analyze_prompt("beautiful anime portrait of a hero")
    assert res["style"] == "anime"
    assert res["orientation"] == "portrait"
    assert res["width"] == 768
    assert res["height"] == 1024
    assert res["steps"] == 30
    assert res["guidance"] == 8.0


def test_analyze_realistic_landscape():
    res = analyze_prompt("realistic landscape photo of mountains")
    assert res["style"] == "realistic"
    assert res["orientation"] == "landscape"
    assert res["width"] == 1024
    assert res["height"] == 768
    assert res["steps"] == 30
    assert res["guidance"] == 6.5


def test_analyze_general_square():
    res = analyze_prompt("abstract shapes")
    assert res["orientation"] == "square"
    assert res["steps"] == 30
    assert res["guidance"] == 7.5

