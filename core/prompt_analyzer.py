from typing import Dict, List


class PromptAnalyzer:
    """Simple prompt analysis for style and orientation."""

    STYLE_KEYWORDS = {
        "anime": ["anime", "manga"],
        "realistic": ["photo", "photorealistic", "realistic"],
        "artistic": ["painting", "artistic", "oil painting", "watercolor"],
        "fantasy": ["fantasy", "magical", "mythical"],
        "cyberpunk": ["cyberpunk", "neon", "futuristic"],
    }

    ORIENTATION_KEYWORDS = {
        "portrait": ["portrait", "vertical", "tall"],
        "landscape": ["landscape", "horizontal", "wide"],
        "square": ["square"],
    }

    ORIENTATION_SETTINGS = {
        "portrait": {"width": 768, "height": 1024},
        "landscape": {"width": 1024, "height": 768},
        "square": {"width": 1024, "height": 1024},
    }

    STYLE_SETTINGS = {
        "anime": {"steps": 35, "guidance": 8.0},
        "realistic": {"steps": 30, "guidance": 6.5},
        "artistic": {"steps": 40, "guidance": 8.5},
        "fantasy": {"steps": 32, "guidance": 7.5},
        "cyberpunk": {"steps": 34, "guidance": 8.0},
    }

    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 7.5

    def detect_styles(self, prompt: str) -> List[str]:
        prompt_l = prompt.lower()
        styles = []
        for style, keywords in self.STYLE_KEYWORDS.items():
            if any(k in prompt_l for k in keywords):
                styles.append(style)
        return styles

    def detect_orientation(self, prompt: str) -> str:
        prompt_l = prompt.lower()
        for orient, keywords in self.ORIENTATION_KEYWORDS.items():
            if any(k in prompt_l for k in keywords):
                return orient
        return "square"

    def analyze(self, prompt: str) -> Dict[str, str | int | float]:
        result: Dict[str, str | int | float] = {}
        styles = self.detect_styles(prompt)
        if styles:
            style = styles[0]
            result["style"] = style
            result.update(self.STYLE_SETTINGS.get(style, {}))
        else:
            result["style"] = "general"
        orient = self.detect_orientation(prompt)
        result["orientation"] = orient
        result.update(self.ORIENTATION_SETTINGS.get(orient, {}))
        result.setdefault("steps", self.DEFAULT_STEPS)
        result.setdefault("guidance", self.DEFAULT_GUIDANCE)
        return result


def analyze_prompt(prompt: str) -> Dict[str, str | int | float]:
    """Convenience wrapper to analyze a prompt."""
    return PromptAnalyzer().analyze(prompt)
