"""Prompt analysis and creative enhancement utilities."""

from typing import Dict, List
import random


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

    STYLE_ENHANCERS = {
        "anime": ["vibrant colors", "cel shading"],
        "realistic": ["photorealistic", "high detail"],
        "artistic": ["painterly", "brush strokes"],
        "fantasy": ["magical atmosphere"],
        "cyberpunk": ["neon lights", "futuristic"],
    }

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


def auto_enhance_prompt(prompt: str) -> str:
    """Automatically add quality enhancers based on detected style."""
    analyzer = PromptAnalyzer()
    styles = analyzer.detect_styles(prompt)
    style = styles[0] if styles else "general"
    enhancers = analyzer.STYLE_ENHANCERS.get(style, [])
    if enhancers:
        return f"{prompt}, {', '.join(enhancers)}"
    return prompt


class CreativePromptEnhancer:
    """Apply fun, themed enhancements to prompts."""

    CREATIVE_MODES = {
        "🎨 Dreamy": {
            "prefix": "ethereal dreamlike",
            "suffix": "soft focus, pastel colors, floating, surreal atmosphere",
            "guidance": 8.5,
            "steps": 35,
        },
        "🌈 Vibrant Pop": {
            "prefix": "bold colorful pop art style",
            "suffix": "bright vivid colors, high contrast, energetic, dynamic composition",
            "guidance": 7.0,
            "steps": 25,
        },
        "🌌 Epic Fantasy": {
            "prefix": "epic fantasy masterpiece",
            "suffix": "magical lighting, dramatic atmosphere, intricate details, award winning",
            "guidance": 9.0,
            "steps": 40,
        },
        "📸 Instant Photo": {
            "prefix": "polaroid photo",
            "suffix": "vintage film aesthetic, nostalgic mood, authentic feel",
            "guidance": 6.0,
            "steps": 20,
        },
        "🎮 Game Art": {
            "prefix": "video game concept art",
            "suffix": "digital painting, professional game art, detailed design",
            "guidance": 7.5,
            "steps": 30,
        },
    }

    SURPRISE_TEMPLATES = [
        "a {adjective} {creature} {doing} in a {location}, {style}",
        "{color} {object} with {magical_property} in {art_style} style",
        "{emotion} {character} surrounded by {element}, {lighting}",
        "portrait of a {adjective} {profession} from {era}, {medium}",
    ]

    WORD_POOLS = {
        "adjective": [
            "whimsical",
            "majestic",
            "tiny",
            "glowing",
            "ancient",
            "futuristic",
            "mystical",
            "cheerful",
            "mysterious",
            "elegant",
        ],
        "creature": [
            "dragon",
            "unicorn",
            "phoenix",
            "griffin",
            "fairy",
            "robot",
            "alien",
            "spirit",
            "elemental",
            "chimera",
        ],
        "doing": [
            "dancing",
            "reading",
            "flying",
            "sleeping",
            "cooking",
            "painting",
            "singing",
            "meditating",
            "exploring",
            "celebrating",
        ],
        "location": [
            "enchanted forest",
            "crystal cave",
            "cloud city",
            "underwater palace",
            "space station",
            "magical library",
            "floating island",
            "neon city",
        ],
        "style": [
            "Studio Ghibli inspired",
            "oil painting",
            "watercolor",
            "digital art",
            "photorealistic",
            "minimalist",
            "art nouveau",
            "cyberpunk aesthetic",
        ],
    }

    def apply_mode(self, prompt: str, mode: str) -> Dict[str, str | int | float]:
        """Return enhanced prompt and settings for the given creative mode."""

        info = self.CREATIVE_MODES.get(mode)
        if not info:
            return {"prompt": prompt}
        enhanced = f"{info['prefix']} {prompt}, {info['suffix']}"
        return {
            "prompt": enhanced,
            "guidance": info.get("guidance", PromptAnalyzer.DEFAULT_GUIDANCE),
            "steps": info.get("steps", PromptAnalyzer.DEFAULT_STEPS),
        }

    def surprise_prompt(self) -> str:
        """Generate a random whimsical prompt using word pools."""

        template = random.choice(self.SURPRISE_TEMPLATES)
        filled = template
        for key, options in self.WORD_POOLS.items():
            filled = filled.replace(f"{{{key}}}", random.choice(options))
        return filled
