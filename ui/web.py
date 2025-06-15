"""
Illustrious AI Studio - Web User Interface

This module creates and manages the main Gradio-based web interface for the AI Studio.
It provides a comprehensive UI for:

CORE FEATURES:
- Image generation with SDXL models
- Interactive chat with Ollama language models
- Image analysis and multimodal interactions
- Gallery management and organization
- Project workspace management
- Model switching and management
- Memory monitoring and optimization

UI COMPONENTS:
- Image Generation Tab: Main creation interface with advanced controls
- Chat Tab: Conversational AI interface with history
- Gallery Tab: Image browsing, filtering, and management
- Model Management Tab: Switch models, monitor status, run tests
- Memory Monitor Tab: Real-time resource monitoring
- Settings Tab: Configuration and preferences

TECHNICAL FEATURES:
- Real-time status updates and progress indicators
- Responsive design with custom CSS styling
- Theme switching (dark/light modes)
- Keyboard shortcuts and accessibility features
- File drag-and-drop support
- Export/import functionality for projects
- Advanced filtering and search capabilities

The interface is built using Gradio's component system and includes custom
JavaScript for enhanced interactivity and user experience.
"""

import json
import logging
import uuid
from pathlib import Path
import os
import subprocess
import sys
import shutil
import random
from datetime import datetime
from typing import List, Tuple, Optional

import gradio as gr

# ==================================================================
# CORE FUNCTIONALITY IMPORTS
# ==================================================================

# Image generation and model management
from core.sdxl import (
    generate_image,
    generate_with_notifications,
    TEMP_DIR,
    get_latest_image,
    init_sdxl,
    get_available_models,
    get_current_model_info,
    test_model_generation,
    switch_sdxl_model,
    PROJECTS_DIR,
    save_to_gallery,
)
from core.image_generator import ImageGenerator

# Configuration and state management
from core.config import CONFIG
from core.state import AppState

# Chat and language model functionality  
from core.ollama import generate_prompt, handle_chat, analyze_image, init_ollama
from core import sdxl, ollama
from core.generation_presets import GENERATION_PRESETS, DEFAULT_PRESET

# Memory and system monitoring
from core.memory import get_model_status, get_memory_stats_markdown, get_memory_stats_wrapper
from core.memory_guardian import (
    start_memory_guardian,
    stop_memory_guardian, 
    get_memory_guardian,
)

# Additional UI utilities
from core.prompt_templates import template_manager
from core.prompt_analyzer import analyze_prompt, PromptAnalyzer, auto_enhance_prompt
from core.gallery_filters import load_gallery_filter, save_gallery_filter

# ==================================================================
# CONSTANTS AND CONFIGURATION
# ==================================================================

logger = logging.getLogger(__name__)

# Theme preference storage
THEME_PREF_FILE = TEMP_DIR / "theme_pref.json"

# Marker for first run modal
FIRST_RUN_FILE = TEMP_DIR / "first_run_complete"

# Special value representing a random seed for image generation
RANDOM_SEED = -1

# Number of images to display per gallery page
ITEMS_PER_PAGE = 20

# Available image resolutions for the generation interface
RESOLUTION_OPTIONS = [
    "512x512 (Square - Fast)",        # Quick generation for testing
    "768x768 (Square - Balanced)",    # Good balance of quality and speed
    "1024x1024 (Square - High Quality)",  # Full quality square images
    "768x512 (Landscape)",            # Standard landscape format
    "512x768 (Portrait)",             # Standard portrait format  
    "1024x768 (Landscape HD)",        # High definition landscape
    "768x1024 (Portrait HD)",         # High definition portrait
]

# Beginner-friendly example prompts to get users started
EXAMPLE_PROMPTS = [
    "A serene landscape with mountains at sunset in watercolor style",
    "Cyberpunk cityscape with neon lights and rainy streets",
    "Fantasy castle surrounded by floating islands in the sky",
    "Portrait of a futuristic astronaut exploring a new planet",
    "Cute anime character holding an umbrella in the rain",
]

# ------------------------------------------------------------------
# Toast Notification Helpers
# ------------------------------------------------------------------

def toast_js(message: str, type_: str = "info") -> str:
    """Return a JS snippet that displays a toast when executed."""
    escaped = message.replace("'", "\\'")
    return f"() => window.showToast('{escaped}', '{type_}')"


def toast_status_js(index: int = 0) -> str:
    """Return JS snippet to display toast from a function output."""
    return f"(...args) => window.notifyStatus(args[{index}] || '')"


# ------------------------------------------------------------------
# Creative prompt helper for simple mode
# ------------------------------------------------------------------

class CreativePromptEnhancer:
    """Make prompt creation magical and fun!"""

    CREATIVE_MODES = {
        "🎨 Dreamy": {
            "prefix": "ethereal dreamlike",
            "suffix": "soft focus, pastel colors, floating, surreal atmosphere",
            "guidance": 8.5,
            "steps": 35,
        },
        "🌈 Vibrant Pop": {
            "prefix": "bold colorful pop art style",
            "suffix": (
                "bright vivid colors, high contrast, energetic, dynamic composition"
            ),
            "guidance": 7.0,
            "steps": 25,
        },
        "🌌 Epic Fantasy": {
            "prefix": "epic fantasy masterpiece",
            "suffix": (
                "magical lighting, dramatic atmosphere, intricate details, award winning"
            ),
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

    SURPRISE_ELEMENTS = {
        "adjectives": [
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
            "cozy",
            "epic",
        ],
        "creatures": [
            "dragon",
            "unicorn",
            "phoenix",
            "griffin",
            "fairy",
            "robot",
            "alien",
            "spirit",
            "owl",
            "fox",
            "butterfly",
            "whale",
        ],
        "actions": [
            "dancing",
            "reading",
            "flying",
            "sleeping",
            "painting",
            "exploring",
            "singing",
            "meditating",
            "playing music",
            "having tea",
        ],
        "locations": [
            "enchanted forest",
            "crystal cave",
            "cloud city",
            "underwater palace",
            "space station",
            "magical library",
            "floating island",
            "cozy cottage",
        ],
        "styles": [
            "Studio Ghibli style",
            "oil painting",
            "watercolor",
            "digital art",
            "stained glass",
            "origami art",
            "chalk art",
            "neon art",
        ],
        "lighting": [
            "golden hour",
            "moonlight",
            "bioluminescent",
            "candlelight",
            "northern lights",
            "sunset",
            "foggy",
            "starlight",
        ],
        "moods": [
            "peaceful",
            "adventurous",
            "whimsical",
            "mysterious",
            "joyful",
            "contemplative",
            "energetic",
            "dreamy",
            "cozy",
            "epic",
        ],
    }

    def get_surprise_prompt(self, base_idea: str = "") -> str:
        """Generate a surprise prompt with random creative elements."""
        if base_idea:
            templates = [
                f"{base_idea} but make it {random.choice(self.SURPRISE_ELEMENTS['adjectives'])} and {random.choice(self.SURPRISE_ELEMENTS['moods'])}",
                f"a {random.choice(self.SURPRISE_ELEMENTS['adjectives'])} {base_idea} in {random.choice(self.SURPRISE_ELEMENTS['locations'])}",
                f"{base_idea} with {random.choice(self.SURPRISE_ELEMENTS['lighting'])} lighting, {random.choice(self.SURPRISE_ELEMENTS['styles'])}",
            ]
            return random.choice(templates)
        return (
            f"a {random.choice(self.SURPRISE_ELEMENTS['adjectives'])} "
            f"{random.choice(self.SURPRISE_ELEMENTS['creatures'])} "
            f"{random.choice(self.SURPRISE_ELEMENTS['actions'])} in "
            f"{random.choice(self.SURPRISE_ELEMENTS['locations'])}, "
            f"{random.choice(self.SURPRISE_ELEMENTS['lighting'])} lighting, "
            f"{random.choice(self.SURPRISE_ELEMENTS['styles'])}"
        )


# ==================================================================
# MAIN GRADIO APP CREATION FUNCTION
# ==================================================================

def create_gradio_app(state: AppState):
    """
    Build and return the complete Gradio web interface.
    
    This is the main entry point for creating the web UI. It constructs
    all tabs, components, and event handlers that make up the interface.
    
    Args:
        state: The shared application state container with model instances
               and runtime data
    
    Returns:
        gr.Blocks: Configured Gradio application ready for launch
        
    The interface includes:
        - Image Generation: SDXL-powered image creation with advanced controls
        - Chat: Interactive conversations with Ollama language models  
        - Gallery: Image browsing, organization, and management
        - Model Management: Switch models, check status, run diagnostics
        - Memory Monitor: Real-time system resource monitoring
        - Settings: Configuration, themes, and preferences
    """
    # Load enhanced CSS styling for modern visual design
    css_file = (Path(__file__).parent / "enhanced.css").read_text()
    css_file += "\n" + (Path(__file__).parent / "custom.css").read_text()

    custom_css = """
    .primary-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-size: 1.2em !important;
        padding: 12px 24px !important;
        transition: transform 0.2s !important;
    }
    .primary-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    .emoji-btn {
        font-size: 1.5em !important;
        padding: 8px 16px !important;
        margin: 4px !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
    }
    .emoji-btn:hover {
        transform: scale(1.1) !important;
        background: rgba(102, 126, 234, 0.1) !important;
    }
    .inspiration-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    """
    css_file += "\n" + custom_css

    enhancer = CreativePromptEnhancer()
    
    # JavaScript functions for loading indicators and UI enhancements
    show_loading_js = "() => { const el = document.querySelector('.loading-indicator'); if(el){ el.style.display='block'; } }"
    hide_loading_js = "() => { const el = document.querySelector('.loading-indicator'); if(el){ el.style.display='none'; } }"

    # Load additional JS utilities (toast system)
    enhanced_js_file = (Path(__file__).parent / "enhanced.js").read_text()
    status_toast_js = toast_status_js(1)
    simple_toast_js = toast_status_js(0)
    
    # ==============================================================
    # THEME MANAGEMENT FUNCTIONS
    # ==============================================================
    
    def load_theme_pref():
        """
        Load user's theme preference from persistent storage.
        
        Returns:
            str: Theme name ('dark' or 'light') or None if not set
        """
        try:
            if THEME_PREF_FILE.exists():
                with open(THEME_PREF_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("theme", "dark")
        except Exception as e:
            logger.error("Error loading theme preference: %s", e)
        return None

    def save_theme_pref(choice: str):
        """
        Save user's theme preference to persistent storage.
        
        Args:
            choice: Theme name to save
            
        Returns:
            str: The saved theme choice
        """
        try:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            with open(THEME_PREF_FILE, "w", encoding="utf-8") as f:
                json.dump({"theme": choice}, f)
        except Exception as e:
            logger.error("Error saving theme preference: %s", e)
        return choice

    # ==============================================================
    # GALLERY MANAGEMENT FUNCTIONS
    # ==============================================================

    # Gallery state variables
    gallery_files: list[Path] = []  # Track file paths for gallery items
    gallery_items: list[tuple[str, str]] = []  # Cached gallery items
    gallery_page: int = 0
    gallery_filter: dict = load_gallery_filter() or {}  # Current filter settings

    def _get_gallery_dir() -> Path:
        """
        Get the appropriate gallery directory based on current project context.
        
        Returns:
            Path: Gallery directory path (project-specific or global)
        """
        if state.current_project:
            return PROJECTS_DIR / state.current_project / "gallery"
        return Path(CONFIG.gallery_dir)

    def load_gallery_items(filters: dict | None = None):
        """
        Load and filter gallery images with metadata.
        
        Scans the gallery directory for images, loads their metadata,
        applies filtering criteria, and returns formatted data for display.
        
        Args:
            filters: Optional filter dictionary to limit results
            
        Returns:
            list: Gallery items formatted for Gradio display
            
        Side effects:
            - Updates global gallery_files list
            - Collects available tags for filtering UI
        """
        nonlocal gallery_files, gallery_items
        if filters is None:
            filters = gallery_filter
            
        gallery_dir = _get_gallery_dir()
        gallery_dir.mkdir(parents=True, exist_ok=True)
        
        items: list[tuple[str, str]] = []  # (image_path, caption) pairs
        gallery_files = []  # Reset file tracking
        tag_set: set[str] = set()  # Collect unique tags
        
        # Scan directory for PNG images (most recent first)
        for img_path in sorted(gallery_dir.glob("*.png"), reverse=True):
            caption = ""
            meta: dict = {}
            meta_path = img_path.with_suffix(".json")  # Metadata file
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    caption = meta.get("prompt", "")
                except Exception as e:  # pragma: no cover - non-critical
                    logger.error("Failed to load metadata for %s: %s", img_path, e)
            tags: list[str] = []
            if isinstance(meta.get("tags"), str):
                tags = [t.strip() for t in meta["tags"].split(",") if t.strip()]
            elif isinstance(meta.get("tags"), list):
                tags = meta["tags"]
            elif meta.get("tag"):
                tag_val = meta["tag"]
                tags = [tag_val] if isinstance(tag_val, str) else tag_val
            tag_set.update(tags)

            tag_filter_val = (filters or {}).get("tag")
            keyword_val = (filters or {}).get("keyword")
            if tag_filter_val and tag_filter_val not in tags:
                continue
            if keyword_val:
                search_text = json.dumps(meta).lower() + " " + caption.lower()
                if keyword_val.lower() not in search_text:
                    continue
            items.append((str(img_path), caption))
            gallery_files.append(img_path)
        gallery_items = items
        return items, sorted(tag_set)

    def load_gallery_page(page: int = 0, per_page: int = ITEMS_PER_PAGE):
        """Load gallery images in chunks."""
        start = page * per_page
        end = start + per_page
        return gallery_items[start:end]

    def refresh_gallery():
        """Update gallery component value and tag dropdown."""
        nonlocal gallery_page
        items, tags = load_gallery_items()
        paged = load_gallery_page(gallery_page)
        return (
            gr.update(value=paged),
            gr.update(choices=tags, value=gallery_filter.get("tag")),
            f"{gallery_page + 1}/{max(1, (len(items) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)}",
        )

    def update_gallery_filters(tag_value: str, keyword_value: str):
        """Persist gallery filters and refresh."""
        nonlocal gallery_filter, gallery_page
        gallery_filter = {"tag": tag_value, "keyword": keyword_value}
        gallery_page = 0
        save_gallery_filter(gallery_filter)
        items, tags = load_gallery_items(gallery_filter)
        paged = load_gallery_page(gallery_page)
        return (
            gr.update(value=paged),
            gr.update(choices=tags, value=tag_value),
            f"{gallery_page + 1}/{max(1, (len(items) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)}",
        )

    def show_metadata(evt: gr.SelectData):
        """Display metadata for selected gallery image."""
        if evt.index is None:
            return {}, ""
        if 0 <= evt.index < len(gallery_files):
            img_path = gallery_files[evt.index]
            meta_path = img_path.with_suffix(".json")
            metadata = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            return metadata, str(img_path)
        return {}, ""

    def open_image_file(path: str):
        """Open the image using the system's default viewer."""
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            return f"Opened {path}"
        except Exception as e:
            return f"Failed to open: {e}"

    def copy_image_path(path: str):
        """Return the path for clipboard copy via JS."""
        return path

    def list_projects():
        PROJECTS_DIR.mkdir(exist_ok=True)
        return [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]

    def set_current_project(name: str):
        state.current_project = name or None
        return refresh_gallery()

    def create_project(name: str):
        name = name.strip()
        if not name:
            return gr.update(), "Please enter a project name"
        (PROJECTS_DIR / name / "gallery").mkdir(parents=True, exist_ok=True)
        state.current_project = name
        return gr.update(choices=list_projects(), value=name), f"Project '{name}' created"

    def delete_project(name: str | None):
        """Delete the selected project directory."""
        if not name:
            return gr.update(), "Please select a project to delete"
        proj_path = PROJECTS_DIR / name
        if not proj_path.exists():
            return gr.update(), "Project not found"
        try:
            shutil.rmtree(proj_path)
        except Exception as e:  # pragma: no cover - unexpected errors
            logger.error("Failed to delete project %s: %s", name, e)
            return gr.update(), f"Failed to delete project: {e}"
        if state.current_project == name:
            state.current_project = None
        return gr.update(choices=list_projects(), value=None), f"Project '{name}' deleted"


    def show_first_run_modal():
        """Display a welcome message on first launch (simplified for Gradio compatibility)."""
        marker = FIRST_RUN_FILE
        
        # Create a simple welcome group instead of modal
        with gr.Group(visible=False) as welcome_group:
            gr.Markdown(
                "### 🎉 Welcome to Illustrious AI Studio\n"
                "Your creative AI workspace is ready! Generate stunning artwork with SDXL and chat with AI assistants."
            )
            close_btn = gr.Button("Get Started", variant="primary")

        def maybe_show():
            if not marker.exists():
                return gr.update(visible=True)
            return gr.update(visible=False)

        def dismiss():
            try:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("done")
            except Exception as e:  # pragma: no cover - unexpected errors
                logger.error("Failed to write first run marker: %s", e)
            return gr.update(visible=False)

        close_btn.click(fn=dismiss, outputs=[welcome_group])
        return maybe_show, welcome_group

    current_theme = load_theme_pref()
    theme_pref_exists = current_theme is not None
    if current_theme is None:
        current_theme = "dark"
    with gr.Blocks(
        title="Illustrious AI Studio",
        theme="default",
        css=css_file
    ) as demo:
        loading_indicator = gr.HTML("<div class='loading-indicator' style='display:none'></div>")
        with gr.Row(elem_classes=["main-container"]):
            with gr.Column():
                gr.Markdown("""
                    # 🎨 Illustrious AI Studio
                    ### Your Creative AI Workspace
                    Transform your imagination into stunning artwork with the power of AI
                """)
                status_display = gr.Markdown(get_model_status(state))
                # Create a hardware info display component that:
                # 1. Shows detected GPU and memory
                # 2. Displays current optimization profile (e.g., "Optimized for RTX 3060 (12GB)")
                # 3. Has a button to "Re-detect Hardware"
                # 4. Shows warnings if hardware is limited
                # 5. Includes tooltips explaining what each optimization does
                # Format as a gr.Markdown component with nice emoji indicators:
                #  - ✅ for good capabilities
                #  - ⚠️ for limitations
                #  - 🚀 for performance tips
                theme_selector = gr.Radio(
                    ["Light", "Dark"],
                    value="Dark" if current_theme == "dark" else "Light",
                    label="Theme",
                    interactive=True,
                    elem_id="theme-selector",
                )
                simple_mode_toggle = gr.Checkbox(
                    value=state.simple_mode,
                    label="Simple Mode",
                    interactive=True,
                    elem_id="simple-mode-toggle",
                )
                with gr.Row():
                    project_selector = gr.Dropdown(
                        label="Project",
                        choices=list_projects(),
                        value=state.current_project,
                        interactive=True,
                    )
                    new_project = gr.Textbox(label="New Project", scale=2)
                    create_project_btn = gr.Button("Create", variant="secondary", size="sm")
                    rename_to = gr.Textbox(label="Rename To", scale=2)
                    rename_project_btn = gr.Button("Rename", variant="secondary", size="sm")
                    delete_project_btn = gr.Button("🗑️", variant="secondary", size="sm")
                project_status = gr.Textbox(label="Project Status", interactive=False, elem_classes=["status-box"], lines=1)

        open_modal_fn, first_run_modal = show_first_run_modal()

        with gr.Tabs():
            if state.simple_mode:
                with gr.Tab("✨ Easy Create"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("## What would you like to create today?")
                            user_idea = gr.Textbox(
                                label="Your idea",
                                placeholder="Try: 'a happy cat', 'sunset beach', 'magical forest'...",
                                lines=2,
                                elem_classes=["primary-input"],
                            )
                            with gr.Group():
                                gr.Markdown("### 🎨 Choose your vibe:")
                                creative_mode = gr.Radio(
                                    choices=list(CreativePromptEnhancer.CREATIVE_MODES.keys()) + ["🎲 Surprise Me!"],
                                    label="",
                                    value="🎨 Dreamy",
                                    elem_classes=["emoji-radio"],
                                )
                            with gr.Group(elem_classes=["inspiration-card"]):
                                gr.Markdown("### 💡 Need inspiration?")
                                inspiration_buttons = []
                                with gr.Row():
                                    for idea in ["cute animal", "magical place", "cool character"]:
                                        btn = gr.Button(f"✨ {idea}", size="sm", elem_classes=["emoji-btn"])
                                        inspiration_buttons.append((btn, idea))
                                with gr.Row():
                                    for idea in ["beautiful landscape", "fantasy creature", "cozy scene"]:
                                        btn = gr.Button(f"✨ {idea}", size="sm", elem_classes=["emoji-btn"])
                                        inspiration_buttons.append((btn, idea))
                                random_prompt_btn = gr.Button(
                                    "🎲 Give me a random idea!",
                                    size="sm",
                                    elem_classes=["emoji-btn"],
                                )
                            generate_btn = gr.Button(
                                "✨ Create Magic! ✨",
                                variant="primary",
                                size="lg",
                                elem_classes=["primary-btn"],
                            )
                            with gr.Accordion("🔧 Advanced Options", open=False):
                                with gr.Row():
                                    custom_steps = gr.Slider(
                                        minimum=10,
                                        maximum=50,
                                        value=25,
                                        step=5,
                                        label="Quality (Higher = Better, but slower)",
                                    )
                                    custom_seed = gr.Number(value=-1, label="Seed (-1 for random)")
                        with gr.Column(scale=2):
                            output_image_simple = gr.Image(label="Your Creation", elem_classes=["output-image"])
                            status_text = gr.Markdown(value="### Ready to create something amazing! 🚀", elem_classes=["status-text"])
                            with gr.Row():
                                remix_btn = gr.Button("🔄 Create Variation", size="sm")
                                enhance_btn = gr.Button("✨ Enhance This", size="sm")
                                save_favorite_btn = gr.Button("❤️ Save as Favorite", size="sm")
                            generation_info = gr.Markdown(visible=False)
                            stats_display = gr.Markdown(
                                """
                                ### 📊 Your Creative Stats
                                🎨 Images Created: 0  
                                ⭐ Favorites: 0  
                                🎯 Current Streak: 0 days
                                """
                            )
                    for btn, idea in inspiration_buttons:
                        btn.click(lambda x=idea: x, outputs=user_idea)

                    def get_random_idea():
                        return enhancer.get_surprise_prompt()

                    random_prompt_btn.click(get_random_idea, outputs=user_idea)

                    def generate_simple(idea, mode, steps, seed, progress=gr.Progress()):
                        if not idea.strip():
                            encouragements = [
                                "💭 What magical thing should we create?",
                                "🎨 Share your creative idea!",
                                "✨ What's in your imagination?",
                                "🌟 Tell me what you'd like to see!",
                            ]
                            return None, random.choice(encouragements), ""

                        progress(0.2, desc="🎨 Preparing your prompt...")

                        if mode == "🎲 Surprise Me!":
                            base_prompt = enhancer.get_surprise_prompt(idea)
                            mode_settings = {"steps": steps or 25, "guidance": 7.5}
                        else:
                            mode_settings = CreativePromptEnhancer.CREATIVE_MODES.get(mode, {})
                            mode_prefix = mode_settings.get("prefix", "")
                            mode_suffix = mode_settings.get("suffix", "")
                            base_prompt = f"{mode_prefix} {idea}, {mode_suffix}".strip()
                            if state.model_status.get("ollama", False):
                                progress(0.3, desc="🤖 Enhancing with AI...")
                                from core.ollama import generate_prompt
                                base_prompt = generate_prompt(state, base_prompt)

                        progress(0.4, desc="🎨 Creating your image...")

                        params = {
                            "prompt": base_prompt,
                            "negative_prompt": "blurry, bad quality, distorted, ugly",
                            "steps": int(steps or mode_settings.get("steps", 25)),
                            "guidance": float(mode_settings.get("guidance", 7.5)),
                            "seed": int(seed) if seed != -1 else -1,
                            "width": 1024,
                            "height": 1024,
                            "save_to_gallery_flag": True,
                        }

                        state.last_generation_params = params

                        # Lazily create the ImageGenerator once per state
                        if not hasattr(state, "image_generator"):
                            state.image_generator = ImageGenerator(state)

                        image, status = state.image_generator.generate(params)
                        if image:
                            success_messages = [
                                "🎉 Wow! Look what you created!",
                                "✨ Amazing! You're a natural artist!",
                                "🌟 Beautiful work! Keep creating!",
                                "🎨 Fantastic! Your creativity is shining!",
                                "🚀 Incredible! You're on fire!",
                                "💫 Magical! This is stunning!",
                                "🌈 Brilliant! What a masterpiece!",
                            ]
                            info = f"""
                            <details>
                            <summary>✨ Creation Details</summary>

                            **Mode:** {mode}  
                            **Prompt:** {base_prompt[:100]}...  
                            **Steps:** {params['steps']}  
                            **Seed:** {params.get('seed', 'Random')}
                            </details>
                            """
                            return image, random.choice(success_messages), info
                        return None, "😅 Oops! Let's try again with different settings...", ""

                    generate_btn.click(
                        generate_simple,
                        inputs=[user_idea, creative_mode, custom_steps, custom_seed],
                        outputs=[output_image_simple, status_text, generation_info],
                    ).then(lambda: gr.update(visible=True), outputs=generation_info)

                    def create_variation(image):
                        if image is None:
                            return None, "🎨 Create an image first!"
                        if not state.last_generation_params:
                            return None, "❌ No generation parameters found"
                        variation_params = state.last_generation_params.copy()
                        variation_modifiers = [
                            "slightly different version of",
                            "alternative take on",
                            "new interpretation of",
                            "fresh perspective on",
                        ]
                        original_prompt = variation_params.get("prompt", "")
                        variation_params["prompt"] = f"{random.choice(variation_modifiers)} {original_prompt}"
                        variation_params["seed"] = -1
                        gen = ImageGenerator(state)
                        new_image, status = gen.generate(variation_params)
                        if new_image:
                            return new_image, "🔄 Created a fresh variation!"
                        return image, "❌ Variation failed, keeping original"

                    remix_btn.click(create_variation, inputs=[output_image_simple], outputs=[output_image_simple, status_text])

                    def enhance_image(image):
                        if image is None:
                            return None, "🎨 Create an image first!"
                        return image, "✨ Enhanced! (Coming soon: upscaling, style transfer, and more!)"

                    enhance_btn.click(enhance_image, inputs=[output_image_simple], outputs=[output_image_simple, status_text])

                with gr.Tab("🎭 Style Explorer"):
                    gr.Markdown("## Explore Different Art Styles\n\nTry your idea in multiple styles at once!")
                    with gr.Row():
                        style_idea = gr.Textbox(label="Your base idea", placeholder="Enter what you want to create...", lines=2)
                        selected_styles = gr.CheckboxGroup(
                            choices=[
                                "🎨 Oil Painting",
                                "💧 Watercolor",
                                "✏️ Pencil Sketch",
                                "🎌 Anime/Manga",
                                "📸 Photography",
                                "🎮 Digital Art",
                            ],
                            label="Select styles to try:",
                            value=["🎨 Oil Painting", "📸 Photography"],
                        )
                    explore_btn = gr.Button("🎨 Create in All Selected Styles", variant="primary")
                    style_gallery = gr.Gallery(label="Style Variations", show_label=True, elem_id="style_gallery", columns=3, rows=2, height="auto")
                    style_status = gr.Markdown()

                    def explore_styles(idea, styles, progress=gr.Progress()):
                        if not idea.strip():
                            return [], "💭 Please enter an idea first!"
                        if not styles:
                            return [], "🎨 Please select at least one style!"
                        results = []
                        style_prompts = {
                            "🎨 Oil Painting": "oil painting, thick brushstrokes, traditional art",
                            "💧 Watercolor": "watercolor painting, soft colors, flowing",
                            "✏️ Pencil Sketch": "detailed pencil sketch, grayscale, artistic",
                            "🎌 Anime/Manga": "anime style, manga art, vibrant colors",
                            "📸 Photography": "professional photography, high resolution, realistic",
                            "🎮 Digital Art": "digital illustration, modern art, crisp details",
                        }
                        total = len(styles)
                        for i, style in enumerate(styles):
                            progress((i + 1) / total, desc=f"Creating {style}...")
                            style_suffix = style_prompts.get(style, "artistic style")
                            full_prompt = f"{idea}, {style_suffix}, high quality, detailed"
                            params = {
                                "prompt": full_prompt,
                                "negative_prompt": "low quality, blurry, distorted",
                                "steps": 20,
                                "guidance": 7.5,
                                "width": 768,
                                "height": 768,
                                "save_to_gallery_flag": True,
                            }
                            gen = ImageGenerator(state)
                            image, _ = gen.generate(params)
                            if image:
                                results.append((image, style))
                        if results:
                            return results, f"✨ Created {len(results)} style variations!"
                        return [], "❌ Generation failed, please try again"

                    explore_btn.click(explore_styles, inputs=[style_idea, selected_styles], outputs=[style_gallery, style_status])
            with gr.Tab("🎨 Create"):
                with gr.Tabs():
                    with gr.Tab("Generate"):
                                    with gr.Row():
                                        with gr.Column():
                                            # Recent prompts functionality
                                            with gr.Row():
                                                recent_prompts = gr.Dropdown(
                                                    label="🕐 Recent Prompts",
                                                    choices=[],
                                                    value=None,
                                                    elem_classes=["dropdown"],
                                                    interactive=True,
                                                    allow_custom_value=False
                                                )
                                                clear_history_btn = gr.Button(
                                                    "🗑️ Clear History",
                                                    variant="secondary",
                                                    size="sm",
                                                    elem_classes=["secondary-button"]
                                                )
                                            with gr.Row():
                                                example_prompts = gr.Dropdown(
                                                    label="🎲 Example Prompts",
                                                    choices=EXAMPLE_PROMPTS,
                                                    value=EXAMPLE_PROMPTS[0] if EXAMPLE_PROMPTS and len(EXAMPLE_PROMPTS) > 0 else None,
                                                    elem_classes=["dropdown"],
                                                    interactive=True,
                                                    allow_custom_value=False
                                                )
                                            prompt = gr.Textbox(
                                                label="Prompt",
                                                placeholder="Describe what you want to create...",
                                                lines=3,
                                                elem_id="prompt-box",
                                                value="",  # Initialize with empty string instead of None
                                            )
                                            
                                            # Quick Style Buttons
                                            with gr.Row():
                                                gr.Markdown("**🎨 Quick Styles:**")
                                            with gr.Row():
                                                quick_styles = [
                                                    {"label": "🌸 Anime", "tooltip": "Applies anime style"},
                                                    {"label": "📷 Realistic", "tooltip": "Applies realistic style"},
                                                    {"label": "🎭 Artistic", "tooltip": "Applies artistic style"},
                                                    {"label": "🧙 Fantasy", "tooltip": "Applies fantasy style"},
                                                    {"label": "🤖 Cyberpunk", "tooltip": "Applies cyberpunk style"},
                                                ]
                                                style_buttons = []
                                                for style in quick_styles:
                                                    btn = gr.Button(
                                                        style["label"],
                                                        variant="secondary",
                                                        size="sm"
                                                    )
                                                    style_buttons.append(btn)
                                                # Assign buttons to named variables for event handlers
                                                assert len(style_buttons) == 5, f"Expected 5 style buttons, but got {len(style_buttons)}"
                                                anime_btn, realistic_btn, artistic_btn, fantasy_btn, cyberpunk_btn = style_buttons
                                            
                                            with gr.Accordion("🎯 Creative Controls", open=False, visible=not state.simple_mode) as creative_controls:
                                                # Model Selection Section
                                                with gr.Group():
                                                    gr.Markdown("### 🎭 Model Selection")
                                                    with gr.Row():
                                                        model_selector = gr.Dropdown(
                                                            label="Art Style Model",
                                                            elem_classes=["dropdown"],
                                                            interactive=True
                                                        )
                                                        test_model_btn = gr.Button(
                                                            "🧪 Test Model",
                                                            variant="secondary",
                                                            size="sm",
                                                            elem_classes=["secondary-button"]
                                                        )
                                                    model_info = gr.Textbox(
                                                        label="Model Information",
                                                        interactive=False,
                                                        lines=1,
                                                        elem_classes=["info-box"]
                                                    )
                                                    model_switch_status = gr.Textbox(
                                                        label="Model Status",
                                                        interactive=False,
                                                        lines=1,
                                                        elem_classes=["status-box"],
                                                        visible=False
                                                    )
                                                with gr.Row():
                                                    preset_selector = gr.Dropdown(
                                                        label="🎛️ Generation Preset",
                                                        choices=list(GENERATION_PRESETS.keys()),
                                                        value="balanced",
                                                        elem_classes=["dropdown"]
                                                    )
                        
                                                with gr.Row():
                                                    steps = gr.Slider(
                                                        10, 100,
                                                        value=30,
                                                        step=1,
                                                        label="Detail Level",
                                                        elem_classes=["artistic-slider"]
                                                    )
                                                    guidance = gr.Slider(
                                                        1.0, 20.0,
                                                        value=7.5,
                                                        step=0.1,
                                                        label="Creative Freedom",
                                                        elem_classes=["artistic-slider"]
                                                    )
                                                with gr.Row():
                                                    resolution = gr.Dropdown(
                                                        label="🖼️ Image Resolution",
                                                        choices=RESOLUTION_OPTIONS,
                                                        value="1024x1024 (Square - High Quality)",
                                                        elem_classes=["dropdown"]
                                                    )
                                                with gr.Accordion("Advanced", open=False):
                                                    negative_prompt = gr.Textbox(
                                                        label="Elements to Avoid",
                                                        value="blurry, low quality, text, watermark, deformed",
                                                        lines=2,
                                                        elem_classes=["textbox"]
                                                    )
                                                    seed = gr.Number(
                                                        value=RANDOM_SEED,
                                                        label="Inspiration Seed (-1 for random)",
                                                        elem_classes=["number-input"]
                                                    )
                                                    save_gallery = gr.Checkbox(
                                                        value=True,
                                                        label="Add to Gallery Collection",
                                                        elem_classes=["checkbox-input"]
                                                    )
                                                    auto_best = gr.Checkbox(
                                                        value=False,
                                                        label="Auto-Best",
                                                        elem_classes=["checkbox-input"]
                                                    )
                                            with gr.Row():
                                                quick_generate_btn = gr.Button(
                                                    "⚡ Quick Generate",
                                                    variant="secondary",
                                                    size="sm",
                                                    elem_classes=["secondary-button"]
                                                )
                                                use_last_btn = gr.Button(
                                                    "🔄 Use Last Settings",
                                                    variant="secondary",
                                                    size="sm",
                                                    elem_classes=["secondary-button"]
                                                )
                                                random_prompt_btn = gr.Button(
                                                    "🎲 Random Prompt",
                                                    variant="secondary",
                                                    size="sm",
                                                    elem_classes=["secondary-button"]
                                                )
                                            with gr.Row():
                                                generate_btn = gr.Button(
                                                    "🎨 Create Masterpiece",
                                                    variant="primary",
                                                    size="lg",
                                                    elem_classes=["primary-button"],
                                                    elem_id="generate-btn",
                                                )
                                                enhance_btn = gr.Button(
                                                    "✨ Enhance Vision",
                                                    variant="secondary",
                                                    elem_classes=["secondary-button"]
                                                )
                                                regenerate_btn = gr.Button(
                                                    "🔄 Regenerate Same",
                                                    variant="secondary",
                                                    size="sm",
                                                    elem_classes=["secondary-button", "regenerate-btn"],
                                                    visible=False
                                                )
                                                reset_btn = gr.Button(
                                                    "♻️ Reset",
                                                    variant="secondary",
                                                    size="sm",
                                                    elem_classes=["secondary-button"]
                                                )
                                        with gr.Column():
                                            output_image = gr.Image(
                                                label="Your Masterpiece",
                                                type="pil",
                                                interactive=False,
                                                elem_classes=["gallery-item"]
                                            )
                                            generation_status = gr.Textbox(
                                                label="Creative Process",
                                                interactive=False,
                                                lines=2,
                                                elem_classes=["status-box"]
                                            )
                                            with gr.Row():
                                                download_btn = gr.DownloadButton(
                                                    "💾 Save Artwork",
                                                    variant="secondary",
                                                    elem_classes=["secondary-button"]
                                                )
                    with gr.Tab("Templates"):
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            gr.Markdown("### 💾 Save Current Prompt")
                                            template_name = gr.Textbox(
                                                label="Template Name",
                                                placeholder="My awesome prompt...",
                                                elem_classes=["textbox"]
                                            )
                                            template_category = gr.Dropdown(
                                                choices=template_manager.templates["categories"],
                                                value="General",
                                                label="Category",
                                                allow_custom_value=True,
                                                elem_classes=["dropdown"]
                                            )
                                            template_tags = gr.Textbox(
                                                label="Tags (comma-separated)",
                                                placeholder="anime, character, fantasy",
                                                elem_classes=["textbox"]
                                            )
                                            save_template_btn = gr.Button(
                                                "💾 Save Template",
                                                variant="primary",
                                                elem_classes=["primary-button"]
                                            )
                                            template_save_status = gr.Textbox(
                                                label="Save Status",
                                                interactive=False,
                                                elem_classes=["status-box"]
                                            )
                                            
                                            gr.Markdown("### 📤 Export/Import")
                                            with gr.Row():
                                                export_btn = gr.Button(
                                                    "📤 Export All",
                                                    variant="secondary",
                                                    elem_classes=["secondary-button"]
                                                )
                                                import_file = gr.File(
                                                    label="Import Templates",
                                                    file_types=[".json"],
                                                    elem_classes=["file-input"]
                                                )
                                            import_merge = gr.Checkbox(
                                                value=True,
                                                label="Merge with existing templates",
                                                elem_classes=["checkbox-input"]
                                            )
                                            export_download = gr.DownloadButton(
                                                "💾 Download Export",
                                                variant="secondary",
                                                visible=False,
                                                elem_classes=["secondary-button"]
                                            )
                                            
                                        with gr.Column(scale=2):
                                            gr.Markdown("### 📋 Template Library")
                                            
                                            # Search and filter
                                            with gr.Row():
                                                template_search = gr.Textbox(
                                                    label="Search Templates",
                                                    placeholder="Search by name, content, or tags...",
                                                    elem_classes=["textbox"]
                                                )
                                                category_filter = gr.Dropdown(
                                                    choices=["All"] + template_manager.templates["categories"],
                                                    value="All",
                                                    label="Filter by Category",
                                                    elem_classes=["dropdown"]
                                                )
                                            
                                            # Template list
                                            template_list = gr.Dropdown(
                                                choices=[],
                                                label="Select Template",
                                                elem_classes=["dropdown"]
                                            )
                                            
                                            # Template preview
                                            with gr.Group():
                                                template_preview_name = gr.Textbox(
                                                    label="Template Name",
                                                    interactive=False,
                                                    elem_classes=["textbox"]
                                                )
                                                template_preview_prompt = gr.Textbox(
                                                    label="Prompt",
                                                    lines=4,
                                                    interactive=False,
                                                    elem_classes=["textbox"]
                                                )
                                                template_preview_negative = gr.Textbox(
                                                    label="Negative Prompt",
                                                    lines=2,
                                                    interactive=False,
                                                    elem_classes=["textbox"]
                                                )
                                                template_preview_info = gr.Textbox(
                                                    label="Template Info",
                                                    lines=2,
                                                    interactive=False,
                                                    elem_classes=["info-box"]
                                                )
                                            
                                            # Template actions
                                            with gr.Row():
                                                use_template_btn = gr.Button(
                                                    "✨ Use Template",
                                                    variant="primary",
                                                    elem_classes=["primary-button"]
                                                )
                                                delete_template_btn = gr.Button(
                                                    "🗑️ Delete",
                                                    variant="secondary",
                                                    elem_classes=["secondary-button"]
                                                )
                                            
                                            # Template statistics
                                            with gr.Accordion("📊 Template Statistics", open=False):
                                                template_stats = gr.JSON(
                                                    label="Collection Stats",
                                                    elem_classes=["json-display"]
                                                )
                                                popular_templates = gr.Dropdown(
                                                    choices=[],
                                                    label="Most Popular Templates",
                                                    elem_classes=["dropdown"]
                                                )
                        
            with gr.Tab("🗂️ Library"):
                            with gr.Row():
                                tag_filter = gr.Dropdown(
                                    label="Filter by Tag",
                                    choices=[],
                                    value=gallery_filter.get("tag"),
                                    allow_custom_value=True,
                                    interactive=True,
                                )
                                keyword_filter = gr.Textbox(
                                    label="Search Metadata",
                                    value=gallery_filter.get("keyword", ""),
                                    interactive=True,
                                )
                            gallery_component = gr.Gallery(
                                label="Gallery",
                                elem_classes=["gallery-section"],
                                columns=[1, 2, 3, 4],
                                height="auto",
                            )
                            metadata_display = gr.JSON(label="Metadata")
                            selected_path = gr.Textbox(visible=False)
                            with gr.Row():
                                open_file_btn = gr.Button(
                                    "📂 Open File",
                                    variant="secondary",
                                    elem_classes=["secondary-button"]
                                )
                                copy_path_btn = gr.Button(
                                    "📋 Copy Path",
                                    variant="secondary",
                                    elem_classes=["secondary-button"]
                                )
                            with gr.Row():
                                prev_page = gr.Button("⬅️", variant="secondary", size="sm")
                                page_display = gr.Textbox(value="1/1", interactive=False, scale=1, label="Page")
                                next_page = gr.Button("➡️", variant="secondary", size="sm")
                            action_status = gr.Textbox(
                                label="Action Status",
                                interactive=False,
                                elem_classes=["status-box"],
                            )
                
            with gr.Tab("🤖 AI Assistant"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    chatbot = gr.Chatbot(
                                        height=500,
                                        show_copy_button=True,
                                        elem_classes=["chatbot"],
                                        type="messages"
                                    )
                                    with gr.Row():
                                        msg = gr.Textbox(
                                            label="Your Creative Brief",
                                            placeholder="Share your artistic vision or use '#generate [description]' to create images...",
                                            scale=4,
                                            elem_classes=["textbox"]
                                        )
                                        send_btn = gr.Button(
                                            "📤 Share",
                                            variant="primary",
                                            scale=1,
                                            elem_classes=["primary-button"]
                                        )
                                    with gr.Row():
                                        clear_btn = gr.Button(
                                            "🗑️ Clear Canvas",
                                            variant="secondary",
                                            elem_classes=["secondary-button"]
                                        )
                                        session_info = gr.Textbox(
                                            value="Session: default",
                                            label="Creative Session",
                                            interactive=False,
                                            scale=2,
                                            elem_classes=["info-box"]
                                        )
                            if state.model_status["multimodal"]:
                                with gr.Group(elem_classes=["image-analysis-group"]):
                                    gr.Markdown(
                                        "### Image Analysis",
                                        elem_classes=["section-header"]
                                    )
                                    with gr.Row():
                                        with gr.Column():
                                            input_image = gr.Image(
                                                label="📁 Upload or Drag & Drop Artwork",
                                                type="pil",
                                                elem_classes=["gallery-item"],
                                                show_download_button=False,
                                                show_share_button=False,
                                                container=True,
                                                sources=["upload", "clipboard"]
                                            )
                                            analysis_question = gr.Textbox(
                                                label="Artistic Inquiry",
                                                value="Describe this artwork in detail",
                                                lines=2,
                                                elem_classes=["textbox"]
                                            )
                                            analyze_btn = gr.Button(
                                                "🔍 Analyze Artwork",
                                                variant="primary",
                                                elem_classes=["primary-button"]
                                            )
                                        with gr.Column(elem_classes=["analysis-container"]):
                                            analysis_output = gr.Textbox(
                                                label="Artistic Analysis",
                                                interactive=False,
                                                lines=15,
                                                show_copy_button=True,
                                                elem_classes=["textbox"]
                                            )
                                gr.Markdown("## ❌ Multimodal Analysis Unavailable")
                                gr.Markdown("Please ensure you have a multimodal LLM and mmproj model configured.")
                                
            with gr.Tab("⚙️ Settings"):
                            gr.Markdown("### Model Configuration")
                            config_display = gr.Code(value=json.dumps(CONFIG.as_dict(), indent=2), language="json", label="Configuration")
                            with gr.Row():
                                sd_model_input = gr.Textbox(
                                    value=CONFIG.sd_model,
                                    label="Artist Model Path",
                                    elem_classes=["textbox"]
                                )
                                ollama_model_input = gr.Textbox(
                                    value=CONFIG.ollama_model,
                                    label="Assistant Model",
                                    elem_classes=["textbox"]
                                )
                            switch_btn = gr.Button(
                                "🔄 Switch Creative Tools",
                                variant="primary",
                                elem_classes=["primary-button"]
                            )
                            refresh_btn = gr.Button(
                                "🔄 Refresh Studio Status",
                                variant="secondary",
                                elem_classes=["secondary-button"]
                            )
                
                            gr.Markdown("### Memory Usage")
                            memory_display = gr.Markdown(get_memory_stats_wrapper(state))
                            refresh_timer = gr.Timer(value=CONFIG.memory_stats_refresh_interval, render=False)
                
                            monitor_status = gr.Textbox(
                                value="",
                                label="Memory Guardian Status",
                                interactive=False,
                                lines=1,
                                elem_classes=["status-box"]
                            )
                            with gr.Row():
                                start_guardian_btn = gr.Button(
                                    "🛡️ Start Guardian",
                                    variant="secondary",
                                    elem_classes=["secondary-button"]
                                )
                                stop_guardian_btn = gr.Button(
                                    "⏹️ Stop Guardian",
                                    variant="secondary",
                                    elem_classes=["secondary-button"]
                                )
                
                            with gr.Row():
                                guardian_cfg = get_memory_guardian(state)
                                profile_dropdown = gr.Dropdown(
                                    choices=["conservative", "balanced", "aggressive"],
                                    value=guardian_cfg.config.get("profile", "balanced"),
                                    label="Memory Profile",
                                )
                                apply_profile_btn = gr.Button(
                                    "Apply Profile",
                                    variant="secondary",
                                    elem_classes=["secondary-button"],
                                )
                
                            with gr.Row():
                                low_slider = gr.Slider(50, 100, value=int(guardian_cfg.thresholds.low_threshold*100), label="Low %")
                                med_slider = gr.Slider(50, 100, value=int(guardian_cfg.thresholds.medium_threshold*100), label="Medium %")
                                high_slider = gr.Slider(50, 100, value=int(guardian_cfg.thresholds.high_threshold*100), label="High %")
                                crit_slider = gr.Slider(50, 100, value=int(guardian_cfg.thresholds.critical_threshold*100), label="Critical %")
                                apply_thresholds_btn = gr.Button(
                                    "Apply Thresholds",
                                    variant="secondary",
                                    elem_classes=["secondary-button"],
                                )
                
                            gr.Markdown("### Model Loader")
                            with gr.Row():
                                sdxl_checkbox = gr.Checkbox(label="SDXL")
                                ollama_checkbox = gr.Checkbox(label="Ollama Text Model")
                                vision_checkbox = gr.Checkbox(label="Vision Model")
                            load_selected_btn = gr.Button(
                                "⚡ Load Selected",
                                variant="secondary",
                                elem_classes=["secondary-button"]
                            )
                            gr.Markdown("### CUDA Memory Management")
                            gr.Markdown(
                                """
                                **Automatic Memory Management Features:**
                                - 🔄 **Auto-retry**: Up to 2 attempts with memory clearing on CUDA OOM errors
                                - 🧹 **Memory clearing**: Automatic CUDA cache clearing before/after generation
                                - 🗑️ **Garbage collection**: Automatic Python garbage collection with memory clearing
                                - ⚡ **Fragmentation prevention**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
                                - 📊 **Smart error handling**: Helpful suggestions when memory limits are reached
                                """
                            )
                            gr.Markdown("### MCP Server")
                            gr.Markdown("**Base URL:** http://localhost:8000")
                            gr.Markdown(
                                """
                                **Available Endpoints:**
                                - `GET /status` - Server status
                                - `POST /generate-image` - Generate images (with automatic memory management)
                                - `POST /chat` - Chat with LLM
                                - `POST /analyze-image` - Analyze images (if multimodal available)
                                """
                            )

        with gr.Row():
            live_status = gr.HTML(
                get_model_status(state).replace("\n", "<br>")
                + "<br>"
                + get_memory_stats_markdown(state).replace("\n", "<br>")
            )

        def update_live_status():
            return (
                get_model_status(state).replace("\n", "<br>")
                + "<br>"
                + get_memory_stats_markdown(state).replace("\n", "<br>")
            )
                        
        # Recent prompts management
        RECENT_PROMPTS_FILE = TEMP_DIR / "recent_prompts.json"
        MAX_RECENT_PROMPTS = 20

        def load_recent_prompts():
            """Load recent prompts from file."""
            try:
                if RECENT_PROMPTS_FILE.exists():
                    with open(RECENT_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                        prompts = json.load(f)
                        return prompts[:MAX_RECENT_PROMPTS]  # Limit to max prompts
                return []
            except Exception as e:
                logger.error(f"Error loading recent prompts: {e}")
                return []

        def save_recent_prompts(prompts):
            """Save recent prompts to file."""
            try:
                TEMP_DIR.mkdir(exist_ok=True)
                with open(RECENT_PROMPTS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(prompts[:MAX_RECENT_PROMPTS], f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error saving recent prompts: {e}")

        def add_to_recent_prompts(prompt_text):
            """Add a prompt to recent prompts list."""
            if not prompt_text or prompt_text.strip() == "":
                return load_recent_prompts()
            
            recent = load_recent_prompts()
            prompt_text = prompt_text.strip()
            
            # Remove if already exists (to move to top)
            recent = [p for p in recent if p != prompt_text]
            
            # Add to beginning
            recent.insert(0, prompt_text)
            
            # Save updated list
            save_recent_prompts(recent)
            return recent

        def clear_recent_prompts():
            """Clear all recent prompts."""
            try:
                if RECENT_PROMPTS_FILE.exists():
                    RECENT_PROMPTS_FILE.unlink()
                return []
            except Exception as e:
                logger.error(f"Error clearing recent prompts: {e}")
                return []


        def select_recent_prompt(selected_prompt):
            """Handle selection of a recent prompt."""
            if selected_prompt:
                return selected_prompt
            return ""
        
        # Model Selection Functions
        def refresh_model_list():
            """Refresh the available models list."""
            try:
                models = get_available_models()
                choices = []
                current_model_path = CONFIG.sd_model
                
                for model in models:
                    # Add status indicators
                    if model["is_current"]:
                        display_text = f"✅ {model['display_name']} ({model['size_mb']}MB) - Current"
                    else:
                        display_text = f"🎭 {model['display_name']} ({model['size_mb']}MB)"
                    
                    choices.append((display_text, model["path"]))
                
                # Find current selection
                current_selection = current_model_path if any(m["path"] == current_model_path for m in models) else None
                
                return gr.update(choices=choices, value=current_selection)
            except Exception as e:
                logger.error("Failed to refresh model list: %s", e)
                return gr.update(
                    choices=[("❌ Error loading models - check the models directory", "")],
                    value="",
                )

        def refresh_project_list():
            return gr.update(choices=list_projects(), value=state.current_project)
        
        def update_model_info(selected_path):
            """Update model information display."""
            if not selected_path:
                return "No model selected"
            
            try:
                models = get_available_models()
                selected_model = next((m for m in models if m["path"] == selected_path), None)
                if not selected_model:
                    return "Model information not available"
                
                info_parts = [
                    f"📁 {selected_model['filename']}",
                    f"💾 {selected_model['size_mb']}MB",
                ]
                
                if selected_model["is_current"]:
                    info_parts.append("✅ Currently Active")
                else:
                    info_parts.append("⏳ Ready to Load")
                
                return " | ".join(info_parts)
                
            except Exception as e:
                return f"❌ Error: {str(e)}. Check logs for details."
        
        def switch_model(selected_path):
            """Switch to the selected model."""
            if not selected_path:
                return (
                    gr.update(visible=True, value="❌ No model selected"),
                    gr.update(),  # model_info
                    gr.update(),   # status_display
                    get_memory_stats_markdown(state)
                )
            
            try:
                # Check if it's already the current model
                if selected_path == CONFIG.sd_model:
                    return (
                        gr.update(visible=True, value="ℹ️ Model already loaded"),
                        update_model_info(selected_path),
                        get_model_status(state),
                        get_memory_stats_markdown(state)
                    )
                
                # Attempt to switch model
                success = switch_sdxl_model(state, selected_path)
                
                if success:
                    status_msg = f"✅ Successfully switched to {Path(selected_path).stem}"
                else:
                    status_msg = (
                        "❌ Failed to switch model. "
                        "Verify the model file exists and check logs for details."
                    )
                
                return (
                    gr.update(visible=True, value=status_msg),
                    update_model_info(selected_path),
                    get_model_status(state),
                    get_memory_stats_markdown(state)
                )
                
            except Exception as e:
                logger.error("Model switch failed: %s", e)
                return (
                    gr.update(
                        visible=True,
                        value=f"❌ Switch failed: {str(e)}. Check logs for details."
                    ),
                    update_model_info(selected_path),
                    get_model_status(state),
                    get_memory_stats_markdown(state)
                )
        
        def test_selected_model(selected_path):
            """Test the selected model with image generation."""
            if not selected_path:
                return (
                    gr.update(visible=True, value="❌ No model selected for testing"),
                    gr.update()
                )
            
            try:
                # Run the test
                success, message, test_image = test_model_generation(state, selected_path)
                
                if success:
                    return (
                        gr.update(visible=True, value=f"✅ {message}"),
                        test_image
                    )
                else:
                    return (
                        gr.update(
                            visible=True,
                            value=f"❌ Test failed: {message}. Verify the model path or check GPU memory."
                        ),
                        gr.update()
                    )
                    
            except Exception as e:
                logger.error("Model test failed: %s", e)
                return (
                    gr.update(
                        visible=True,
                        value=f"❌ Test error: {str(e)}. Check logs for more details."
                    ),
                    gr.update()
                )
        
        # Event Handlers
        theme_selector.change(
            fn=lambda m: save_theme_pref(m.lower()) if m else None,
            inputs=theme_selector,
            outputs=[],
            js="(mode) => { if(mode === 'Dark'){ document.documentElement.classList.add('dark'); } else { document.documentElement.classList.remove('dark'); } }"
        )
        def toggle_simple(val: bool):
            state.simple_mode = val
            return gr.update(visible=not val)

        simple_mode_toggle.change(
            fn=toggle_simple,
            inputs=simple_mode_toggle,
            outputs=[creative_controls]
        )
        project_selector.change(
            fn=set_current_project,
            inputs=project_selector,
            outputs=[gallery_component, tag_filter, page_display]
        )

        create_project_btn.click(
            fn=create_project,
            inputs=new_project,
            outputs=[project_selector, project_status]
        ).then(
            fn=lambda: refresh_gallery(),
            outputs=[gallery_component, tag_filter, page_display]
        )
        rename_project_btn.click(
            fn=lambda old, new: rename_project(state, old, new),
            inputs=[project_selector, rename_to],
            outputs=[project_selector, project_status]
        ).then(
            fn=lambda: refresh_gallery(),
            outputs=[gallery_component, tag_filter, page_display]
        )
        try:
            delete_project_btn.click(
                fn=delete_project,
                inputs=project_selector,
                outputs=[project_selector, project_status],
                js="(p)=>confirm('Delete project ' + p + '?') ? p : null"
            ).then(
                fn=lambda: refresh_gallery(),
                outputs=[gallery_component, tag_filter, page_display]
            )
        except TypeError:  # For dummy components in tests
            delete_project_btn.click(
                fn=delete_project,
                inputs=project_selector,
                outputs=[project_selector, project_status]
            ).then(
                fn=lambda: refresh_gallery(),
                outputs=[gallery_component, tag_filter, page_display]
            )
        enhance_btn.click(fn=lambda p: generate_prompt(state, p), inputs=prompt, outputs=prompt)
        
        # Encapsulate generation state in a dedicated class for maintainability
        class GenerationState:
            def __init__(self):
                self.prompt = ""
                self.negative_prompt = "blurry, low quality, text, watermark, deformed"
                self.steps = 30
                self.guidance = 7.5
                self.seed = RANDOM_SEED
                self.save_gallery = True
                self.resolution = "1024x1024 (Square - High Quality)"
                self.auto_best = False

        generation_state = GenerationState()

        def update_prompt_value(value):
            generation_state.prompt = value or ""
            # No return value needed as outputs=[] in .change()

        def update_negative_prompt_value(value):
            generation_state.negative_prompt = value or ""
            # No return value needed

        def update_steps_value(value):
            generation_state.steps = value if value is not None else 30
            # No return value needed

        def update_guidance_value(value):
            generation_state.guidance = value if value is not None else 7.5
            # No return value needed

        def update_seed_value(value):
            generation_state.seed = value if value is not None else RANDOM_SEED
            # No return value needed

        def update_save_gallery_value(value):
            generation_state.save_gallery = value if value is not None else True
            # No return value needed

        def update_resolution_value(value):
            generation_state.resolution = value or "1024x1024 (Square - High Quality)"
            # No return value needed

        def update_auto_best_value(value):
            generation_state.auto_best = value if value is not None else False
            # No return value needed

        # State-based generation function that doesn't rely on parameter passing
        async def generate_and_update_history(progress=gr.Progress()):
            try:
                # Get values from our encapsulated generation state
                logger.info("UI: Using state-based parameter retrieval")
                
                p = generation_state.prompt.strip() if generation_state.prompt else "A beautiful landscape with mountains and trees"
                n = generation_state.negative_prompt.strip() if generation_state.negative_prompt else "blurry, low quality, text, watermark, deformed"
                st = generation_state.steps
                g = generation_state.guidance
                se = generation_state.seed
                save_flag = generation_state.save_gallery
                res = generation_state.resolution
                auto_flag = generation_state.auto_best
                
                logger.info(f"UI: Using parameters - prompt='{p[:50]}...', steps={st}, guidance={g}, seed={se}")
                
                # Convert to string if needed
                if not isinstance(p, str):
                    try:
                        p = str(p)
                    except Exception as e:
                        return (
                            None,
                            f"❌ Error: Cannot convert prompt to string: {e}",
                            gr.update(),
                            gr.update(visible=False)
                        )
                
                # Clean the prompt and check if empty
                p = p.strip()
                if not p:
                    return (
                        None, 
                        "❌ Error: Prompt cannot be empty. Please provide a descriptive text prompt.",
                        gr.update(),
                        gr.update(visible=False)
                    )
                
                logger.info(f"UI: Cleaned prompt: '{p[:50]}...' (length: {len(p)})")
                
                # Validate negative prompt
                if n is None:
                    n = ""
                elif not isinstance(n, str):
                    logger.warning(f"Invalid negative prompt type: {type(n)}, using empty string")
                    n = ""
                n = n.strip()
                
                if auto_flag:
                    analysis = analyze_prompt(p)
                    st = analysis.get("steps", st) if analysis else st
                    g = analysis.get("guidance", g) if analysis else g
                    width = analysis.get("width", 1024) if analysis else 1024
                    height = analysis.get("height", 1024) if analysis else 1024
                else:
                    width, height = parse_resolution(res)
                
                # Enhanced parameter validation and type conversion
                try:
                    st = int(st) if st is not None else 30
                    if st <= 0 or st > 200:
                        logger.warning(f"UI: Steps {st} out of range, using 30")
                        st = 30
                except (ValueError, TypeError):
                    logger.warning(f"UI: Invalid steps '{st}', using 30")
                    st = 30
                
                try:
                    g = float(g) if g is not None else 7.5
                    if g <= 0 or g > 50:
                        logger.warning(f"UI: Guidance {g} out of range, using 7.5")
                        g = 7.5
                except (ValueError, TypeError):
                    logger.warning(f"UI: Invalid guidance '{g}', using 7.5")
                    g = 7.5
                
                try:
                    width = int(width) if width is not None else 1024
                    height = int(height) if height is not None else 1024
                    if width <= 0 or width > 2048 or height <= 0 or height > 2048:
                        logger.warning(f"UI: Dimensions {width}x{height} out of range, using 1024x1024")
                        width, height = 1024, 1024
                except (ValueError, TypeError):
                    logger.warning(f"UI: Invalid dimensions '{width}x{height}', using 1024x1024")
                    width, height = 1024, 1024
                
                # Validate and sanitize seed parameter
                try:
                    if se is None:
                        se = RANDOM_SEED  # Use random seed
                    elif se == "" or se == "":
                        se = RANDOM_SEED  # Empty string means random
                    else:
                        se = int(float(se))  # Convert to int, handling potential float strings
                        # Clamp to valid range
                        if se < -1 or se >= 2**32:
                            logger.warning(f"UI: Seed {se} out of range, using random seed")
                            se = RANDOM_SEED
                except (ValueError, TypeError) as e:
                    logger.warning(f"UI: Invalid seed '{se}': {e}, using random seed")
                    se = RANDOM_SEED
                
                logger.info(f"UI: Validated parameters - prompt='{p[:50]}...', steps={st}, guidance={g}, seed={se}, dimensions={width}x{height}")
                
                # Generate the image with resolution
                params = {
                    "prompt": p,
                    "negative_prompt": n,
                    "steps": st,
                    "guidance": g,
                    "seed": se,
                    "save_to_gallery_flag": save_flag,
                    "width": width,
                    "height": height,
                }
                image, status = await generate_with_notifications(state, params, progress)
                
                # Store parameters for regenerate functionality if generation was successful
                if image is not None:
                    state.last_generation_params = {
                        "prompt": p,
                        "negative_prompt": n,
                        "steps": st,
                        "guidance": g,
                        "seed": se,
                        "save_gallery": save_flag,
                        "resolution": res,
                        "width": width,
                        "height": height
                    }
                
                # Add to recent prompts if generation was successful and prompt is not empty
                if p and p.strip() and image is not None:
                    updated_prompts = add_to_recent_prompts(p.strip())
                    regenerate_visible = True
                    return image, status, gr.update(choices=updated_prompts), gr.update(visible=regenerate_visible)
                else:
                    regenerate_visible = image is not None and state.last_generation_params is not None
                    return image, status, gr.update(), gr.update(visible=regenerate_visible)
                    
            except Exception as e:
                logger.error(f"UI parameter validation failed: {e}")
                return (
                    None,
                    f"❌ Error: Parameter validation failed: {str(e)}. Please check your inputs and try again.",
                    gr.update(),
                    gr.update(visible=False)
                )
        
        # Regenerate function
        async def regenerate_image(progress=gr.Progress()):
            if state.last_generation_params is None:
                return (
                    None,
                    "❌ No previous generation to repeat. Generate an image first.",
                    gr.update(),
                )

            params = state.last_generation_params
            image, status = await generate_with_notifications(
                state,
                {
                    "prompt": params["prompt"],
                    "negative_prompt": params["negative_prompt"],
                    "steps": params["steps"],
                    "guidance": params["guidance"],
                    "seed": params["seed"],
                    "save_to_gallery_flag": params["save_to_gallery_flag"],
                    "width": params["width"],
                    "height": params["height"],
                },
                progress,
            )
            
            if image is not None and params["prompt"].strip():
                updated_prompts = add_to_recent_prompts(params["prompt"].strip())
                return image, status, gr.update(choices=updated_prompts)
            else:
                return image, status, gr.update()

        async def quick_generate_image(progress=gr.Progress()):
            # Use encapsulated generation state for prompt and negative prompt
            p = generation_state.prompt.strip() if generation_state.prompt else "A beautiful landscape with mountains and trees"
            n = generation_state.negative_prompt.strip() if generation_state.negative_prompt else "blurry, low quality, text, watermark, deformed"
            
            preset = GENERATION_PRESETS.get(DEFAULT_PRESET, {})
            st = preset.get("steps", 25)
            g = preset.get("guidance", 7.5)
            width = preset.get("width", 768)
            height = preset.get("height", 768)
            res_str = get_resolution_option(width, height) # Store the string for state
            
            logger.info(f"UI (Quick Generate): Using parameters - prompt='{p[:50]}...', steps={st}, guidance={g}, seed={RANDOM_SEED}")
            
            if not p:
                return (
                    None, 
                    "❌ Error: Prompt cannot be empty for Quick Generate.",
                    gr.update(),
                    gr.update(visible=False)
                )

            # Create proper GenerationParams
            gen_params = {
                "prompt": p,
                "negative_prompt": n,
                "steps": int(st),
                "guidance": float(g),
                "seed": RANDOM_SEED,
                "save_to_gallery_flag": True, # Default for quick generate
                "width": int(width),
                "height": int(height),
            }
            image, status = await generate_with_notifications(state, gen_params, progress)

            if image is not None:
                state.last_generation_params = {
                    "prompt": p, "negative_prompt": n, "steps": st, "guidance": g, "seed": RANDOM_SEED,
                    "save_to_gallery_flag": True, "resolution": res_str, "width": width, "height": height
                }
            
            if p and p.strip() and image is not None:
                updated_prompts = add_to_recent_prompts(p.strip())
                return image, status, gr.update(choices=updated_prompts), gr.update(visible=True)
            else:
                return image, status, gr.update(), gr.update(visible=image is not None)

        def restore_last_settings(curr_p, curr_n, curr_st, curr_g, curr_se, curr_res):
            params = state.last_generation_params
            if params is None:
                return (
                    curr_p,
                    curr_n,
                    curr_st,
                    curr_g,
                    curr_se,
                    curr_res,
                    gr.update(visible=False),
                )
            return (
                params.get("prompt", curr_p),
                params.get("negative_prompt", curr_n),
                params.get("steps", curr_st),
                params.get("guidance", curr_g),
                params.get("seed", curr_se),
                params.get(
                    "resolution",
                    get_resolution_option(params.get("width", 1024), params.get("height", 1024)),
                ),
                gr.update(visible=True),
            )

        def get_random_prompt():
            return random.choice(EXAMPLE_PROMPTS)

        try:
            generate_btn.click(
                fn=generate_and_update_history,
                inputs=[],
                outputs=[output_image, generation_status, recent_prompts, regenerate_btn],
                js=show_loading_js,
            ).then(
                js=status_toast_js
            ).then(
                fn=refresh_gallery,
                inputs=None,
                outputs=[gallery_component, tag_filter, page_display],
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            ).then(
                js=hide_loading_js
            )
        except TypeError:
            generate_btn.click(
                fn=generate_and_update_history,
                inputs=[],
                outputs=[output_image, generation_status, recent_prompts, regenerate_btn],
            ).then(
                js=status_toast_js
            ).then(
                fn=refresh_gallery,
                inputs=None,
                outputs=[gallery_component, tag_filter, page_display],
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            )

        try:
            regenerate_btn.click(
                fn=regenerate_image,
                inputs=[],
                outputs=[output_image, generation_status, recent_prompts],
                js=show_loading_js,
            ).then(
                js=status_toast_js
            ).then(
                fn=refresh_gallery,
                inputs=None,
                outputs=[gallery_component, tag_filter, page_display],
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            ).then(
                js=hide_loading_js
            )
        except TypeError:
            regenerate_btn.click(
                fn=regenerate_image,
                inputs=[],
                outputs=[output_image, generation_status, recent_prompts],
            ).then(
                js=status_toast_js
            ).then(
                fn=refresh_gallery,
                inputs=None,
                outputs=[gallery_component, tag_filter, page_display],
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            )

        try:
            quick_generate_btn.click(
                fn=quick_generate_image, # No inputs needed as it uses global state
                inputs=[],
                outputs=[output_image, generation_status, recent_prompts, regenerate_btn],
                js=show_loading_js,
            ).then(
                js=status_toast_js
            ).then(
                fn=refresh_gallery,
                inputs=None,
                outputs=[gallery_component, tag_filter, page_display],
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            ).then(
                js=hide_loading_js
            )
        except TypeError:
            quick_generate_btn.click(
                fn=quick_generate_image, # No inputs needed
                inputs=[],
                outputs=[output_image, generation_status, recent_prompts, regenerate_btn],
            ).then(
                js=status_toast_js
            ).then(
                fn=refresh_gallery,
                inputs=None,
                outputs=[gallery_component, tag_filter, page_display],
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            )

        use_last_btn.click(
            fn=restore_last_settings,
            inputs=[prompt, negative_prompt, steps, guidance, seed, resolution],
            outputs=[prompt, negative_prompt, steps, guidance, seed, resolution, regenerate_btn],
        )

        random_prompt_btn.click(
            fn=get_random_prompt,
            inputs=None,
            outputs=[prompt],
        )

        # Reset controls handler
        def reset_generation():
            state.last_generation_params = None
            defaults = CONFIG.generation_defaults or {}
            steps_def = defaults.get("steps", 30)
            guidance_def = defaults.get("guidance_scale", 7.5)
            width_def = defaults.get("width", 1024)
            height_def = defaults.get("height", 1024)
            resolution_def = get_resolution_option(width_def, height_def)
            return (
                "",
                defaults.get("negative_prompt", "blurry, low quality, text, watermark, deformed"),
                steps_def,
                guidance_def,
                RANDOM_SEED,
                resolution_def,
                gr.update(visible=False)
            )

        reset_btn.click(
            fn=reset_generation,
            inputs=None,
            outputs=[
                prompt,
                negative_prompt,
                steps,
                guidance,
                seed,
                resolution,
                regenerate_btn,
            ],
        )


        def apply_generation_preset(name: str):
            preset = GENERATION_PRESETS.get(name, GENERATION_PRESETS.get("balanced"))
            steps_val = preset.get("steps", 30)
            guidance_val = preset.get("guidance", 7.5)
            res_option = get_resolution_option(preset.get("width", 1024), preset.get("height", 1024))
            return steps_val, guidance_val, res_option

        preset_selector.change(
            fn=apply_generation_preset,
            inputs=[preset_selector],
            outputs=[steps, guidance, resolution],
        )

        # Recent prompts selection handler
        recent_prompts.change(
            fn=select_recent_prompt,
            inputs=[recent_prompts],
            outputs=[prompt]
        )

        # Example prompts selection handler
        # Pass the selected example prompt directly to the output. This lambda serves as a placeholder for potential future processing.
        example_prompts.change(
            fn=lambda p: p,
            inputs=[example_prompts],
            outputs=[prompt]
        )
        
        # Clear history handler
        def clear_history():
            clear_recent_prompts()
            return gr.update(choices=[])
        
        clear_history_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[recent_prompts]
        )
        
        # Initialize recent prompts on load
        def init_recent_prompts():
            return gr.update(choices=load_recent_prompts())
        
        demo.load(
            fn=init_recent_prompts,
            inputs=[],
            outputs=[recent_prompts]
        )

        demo.load(
            js=f"""
() => {{
  const hasPref = {str(theme_pref_exists).lower()};
  let mode;
  if(hasPref) {{
    mode = '{current_theme}';
  }} else {{
    mode = 'dark';
  }}
  if(mode === 'dark') {{
    document.documentElement.classList.add('dark');
  }} else {{
    document.documentElement.classList.remove('dark');
  }}
}}
"""
        )

        demo.load(
            js="""
            () => {
                const box = document.getElementById('prompt-box');
                const btn = document.getElementById('generate-btn');
                if (box && btn) {
                    box.addEventListener('keydown', (e) => {
                        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                            btn.click();
                        }
                    });
                }
            }
            """
        )

        # Load toast notification utilities
        demo.load(js=enhanced_js_file)

        def prepare_download(image):
            if image is None:
                return None
            temp_path = TEMP_DIR / f"download_{uuid.uuid4().hex[:8]}.png"
            image.save(temp_path)
            return str(temp_path)

        generate_btn.click(fn=prepare_download, inputs=output_image, outputs=download_btn)

        def chat_wrapper(message, history):
            if not message.strip():
                return history or [], ""
            result_history, empty_msg = handle_chat(state, message, session_id="default", chat_history=history)
            return result_history, ""

        def chat_wrapper_with_image_update(message, history):
            result_history, empty_msg = chat_wrapper(message, history)
            if message.lower().startswith("#generate") or "generate image" in message.lower():
                return result_history, empty_msg, get_latest_image(state)
            return result_history, empty_msg, gr.update(value=None)

        send_btn.click(
            fn=chat_wrapper_with_image_update,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, output_image],
        ).then(
            fn=update_live_status,
            inputs=None,
            outputs=[live_status],
        )
        msg.submit(
            fn=chat_wrapper_with_image_update,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, output_image],
        ).then(
            fn=update_live_status,
            inputs=None,
            outputs=[live_status],
        )
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        if state.model_status["multimodal"]:
            analyze_btn.click(
                fn=lambda img, q: analyze_image(state, img, q),
                inputs=[input_image, analysis_question],
                outputs=analysis_output,
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            )
        
        # Model Selection Event Handlers
        try:
            model_selector.change(
                fn=update_model_info,
                inputs=model_selector,
                outputs=model_info,
                js=show_loading_js
            ).then(
                fn=switch_model,
                inputs=model_selector,
                outputs=[model_switch_status, model_info, status_display, memory_display]
            ).then(
                js=hide_loading_js
            )
        except TypeError:
            model_selector.change(
                fn=update_model_info,
                inputs=model_selector,
                outputs=model_info,
            ).then(
                fn=switch_model,
                inputs=model_selector,
                outputs=[model_switch_status, model_info, status_display, memory_display]
            )
        
        test_model_btn.click(
            fn=test_selected_model,
            inputs=model_selector,
            outputs=[model_switch_status, output_image]
        )
        
        # Prompt Template Management Functions
        def refresh_template_list(category_filter_value="All", search_query=""):
            """Refresh the template list based on filters."""
            if search_query:
                templates = template_manager.search_templates(search_query)
            elif category_filter_value != "All":
                templates = template_manager.get_templates_by_category(category_filter_value)
            else:
                templates = template_manager.get_templates_by_category()
            
            choices = []
            for template in templates:
                usage_info = f" (used {template.get('usage_count', 0)} times)" if template.get('usage_count', 0) > 0 else ""
                choices.append((f"{template['name']}{usage_info}", template['id']))
            
            return gr.update(choices=choices, value=None)
        
        def load_template_preview(template_id):
            """Load template details for preview."""
            if not template_id:
                return "", "", "", ""
            
            template = template_manager.get_template(template_id)
            if not template:
                return "", "", "", "Template not found"
            
            info_text = f"Category: {template['category']} | Tags: {', '.join(template['tags'])} | Created: {template['created_at'][:10]} | Used: {template.get('usage_count', 0)} times"
            
            return (
                template['name'],
                template['prompt'],
                template['negative_prompt'],
                info_text
            )
        
        def save_current_template(name, category, tags, current_prompt, current_negative):
            """Save current prompt as a template."""
            if not name or not current_prompt:
                return "Please enter a template name and ensure you have a prompt to save."
            
            try:
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
                template_id = template_manager.add_template(
                    name=name,
                    prompt=current_prompt,
                    negative_prompt=current_negative or "",
                    category=category,
                    tags=tag_list
                )
                return f"✅ Template '{name}' saved successfully! (ID: {template_id[:8]}...)"
            except Exception as e:
                return (
                    f"❌ Failed to save template: {str(e)}. "
                    "Check write permissions in the templates directory."
                )
        
        def use_selected_template(template_id):
            """Apply selected template to the main prompt fields."""
            if not template_id:
                return "", "", "Please select a template first"
            
            template = template_manager.get_template(template_id)
            if not template:
                return "", "", "Template not found"
            
            # Increment usage count
            template_manager.increment_usage(template_id)
            
            return (
                template['prompt'],
                template['negative_prompt'],
                f"✅ Applied template '{template['name']}'"
            )
        
        def delete_selected_template(template_id):
            """Delete the selected template."""
            if not template_id:
                return "Please select a template to delete"
            
            template = template_manager.get_template(template_id)
            if not template:
                return "Template not found"
            
            if template_manager.delete_template(template_id):
                return f"✅ Template '{template['name']}' deleted successfully"
            else:
                return "❌ Failed to delete template. Check permissions."
        
        def export_templates():
            """Export all templates to a file."""
            try:
                export_path = template_manager.export_templates()
                return gr.update(value=export_path, visible=True)
            except Exception as e:
                return gr.update(visible=False)
        
        def import_templates(file_obj, merge_flag):
            """Import templates from uploaded file."""
            if file_obj is None:
                return "Please select a file to import"
            
            try:
                import_path = Path(file_obj.name)
                count = template_manager.import_templates(import_path, merge=merge_flag)
                return f"✅ Successfully imported {count} templates!"
            except Exception as e:
                return (
                    f"❌ Import failed: {str(e)}. "
                    "Verify the file format and check logs for details."
                )
        
        def get_template_statistics():
            """Get template collection statistics."""
            stats = template_manager.get_template_stats()
            popular = template_manager.get_popular_templates(5)
            popular_choices = [(f"{t['name']} ({t.get('usage_count', 0)} uses)", t['id']) for t in popular]
            
            return stats, gr.update(choices=popular_choices)
        
        # Template event handlers
        template_search.change(
            fn=lambda search, category: refresh_template_list(category, search),
            inputs=[template_search, category_filter],
            outputs=template_list
        )
        
        category_filter.change(
            fn=lambda category, search: refresh_template_list(category, search),
            inputs=[category_filter, template_search],
            outputs=template_list
        )
        
        template_list.change(
            fn=load_template_preview,
            inputs=template_list,
            outputs=[template_preview_name, template_preview_prompt, template_preview_negative, template_preview_info]
        )
        
        save_template_btn.click(
            fn=save_current_template,
            inputs=[template_name, template_category, template_tags, prompt, negative_prompt],
            outputs=template_save_status
        ).then(
            js=simple_toast_js
        ).then(
            fn=lambda: refresh_template_list(),
            outputs=template_list
        )
        
        use_template_btn.click(
            fn=use_selected_template,
            inputs=template_list,
            outputs=[prompt, negative_prompt, generation_status]
        )
        
        delete_template_btn.click(
            fn=delete_selected_template,
            inputs=template_list,
            outputs=template_save_status
        ).then(
            js=simple_toast_js
        ).then(
            fn=lambda: refresh_template_list(),
            outputs=template_list
        )
        
        export_btn.click(
            fn=export_templates,
            outputs=export_download
        ).then(
            js=toast_js("Templates exported", "success")
        )
        
        import_file.change(
            fn=import_templates,
            inputs=[import_file, import_merge],
            outputs=template_save_status
        ).then(
            js=simple_toast_js
        ).then(
            fn=lambda: refresh_template_list(),
            outputs=template_list
        )

        tag_filter.change(
            fn=update_gallery_filters,
            inputs=[tag_filter, keyword_filter],
            outputs=[gallery_component, tag_filter, page_display]
        )

        keyword_filter.change(
            fn=update_gallery_filters,
            inputs=[tag_filter, keyword_filter],
            outputs=[gallery_component, tag_filter, page_display]
        )

        if hasattr(gallery_component, "select"):
            gallery_component.select(
                fn=show_metadata,
                inputs=None,
                outputs=[metadata_display, selected_path],
            )
        open_file_btn.click(
            fn=open_image_file,
            inputs=selected_path,
            outputs=action_status,
        )
        try:
            copy_path_btn.click(
                fn=copy_image_path,
                inputs=selected_path,
                outputs=action_status,
                js="(p)=>{navigator.clipboard.writeText(p); return 'Copied';}"
            )
        except TypeError:  # For dummy components in tests
            copy_path_btn.click(
                fn=copy_image_path,
                inputs=selected_path,
                outputs=action_status,
            )

        def next_page_fn():
            nonlocal gallery_page
            items, _ = load_gallery_items(gallery_filter)
            total_pages = max(1, (len(items) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
            gallery_page = min(gallery_page + 1, total_pages - 1)
            paged = load_gallery_page(gallery_page)
            return gr.update(value=paged), f"{gallery_page + 1}/{total_pages}"

        def prev_page_fn():
            nonlocal gallery_page
            gallery_page = max(gallery_page - 1, 0)
            items, _ = load_gallery_items(gallery_filter)
            total_pages = max(1, (len(items) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
            paged = load_gallery_page(gallery_page)
            return gr.update(value=paged), f"{gallery_page + 1}/{total_pages}"

        prev_page.click(
            fn=prev_page_fn,
            outputs=[gallery_component, page_display],
        )

        next_page.click(
            fn=next_page_fn,
            outputs=[gallery_component, page_display],
        )

        def get_monitor_status():
            guardian = get_memory_guardian(state)
            return "🟢 Monitoring" if guardian.is_monitoring else "🔴 Stopped"

        def start_guardian_ui():
            start_memory_guardian(state)
            return get_monitor_status(), get_memory_stats_markdown(state)

        def stop_guardian_ui():
            stop_memory_guardian(state)
            return get_monitor_status(), get_memory_stats_markdown(state)

        def set_profile_ui(profile):
            guardian = get_memory_guardian(state)
            try:
                guardian.set_profile(profile)
            except ValueError as e:
                logger.error("%s", e)
            return get_memory_stats_markdown(state)

        def set_thresholds_ui(low, medium, high, critical):
            guardian = get_memory_guardian(state)
            for lvl, val in [
                ("low", low),
                ("medium", medium),
                ("high", high),
                ("critical", critical),
            ]:
                try:
                    guardian.set_threshold(lvl, val)
                except ValueError as e:
                    logger.error("%s", e)
            return get_memory_stats_markdown(state)
        
        def do_switch(sd_path, ollama_name):
            if sd_path:
                sdxl.switch_sdxl_model(state, sd_path)
            if ollama_name:
                ollama.switch_ollama_model(state, ollama_name)
            return (
                get_model_status(state),
                json.dumps(CONFIG.as_dict(), indent=2),
                get_memory_stats_markdown(state),
            )
        try:
            switch_btn.click(
                fn=do_switch,
                inputs=[sd_model_input, ollama_model_input],
                outputs=[status_display, config_display, memory_display, monitor_status],
                js=show_loading_js,
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            ).then(
                js=hide_loading_js
            )
        except TypeError:
            switch_btn.click(
                fn=do_switch,
                inputs=[sd_model_input, ollama_model_input],
                outputs=[status_display, config_display, memory_display, monitor_status],
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            )
        refresh_btn.click(
            fn=lambda: (
                get_model_status(state),
                get_memory_stats_markdown(state),
                get_monitor_status(),
            ),
            outputs=[status_display, memory_display, monitor_status],
        ).then(
            fn=update_live_status,
            inputs=None,
            outputs=[live_status],
        )

        def load_selected_models(load_s, load_o, load_v):
            if load_s:
                sdxl.init_sdxl(state)
            if load_o or load_v:
                ollama.init_ollama(state)
            return (
                get_model_status(state),
                get_memory_stats_markdown(state),
                get_monitor_status(),
            )

        try:
            load_selected_btn.click(
                fn=load_selected_models,
                inputs=[sdxl_checkbox, ollama_checkbox, vision_checkbox],
                outputs=[status_display, memory_display, monitor_status],
                js=show_loading_js,
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            ).then(
                js=hide_loading_js
            )
        except TypeError:
            load_selected_btn.click(
                fn=load_selected_models,
                inputs=[sdxl_checkbox, ollama_checkbox, vision_checkbox],
                outputs=[status_display, memory_display, monitor_status],
            ).then(
                fn=update_live_status,
                inputs=None,
                outputs=[live_status],
            )

        start_guardian_btn.click(
            fn=start_guardian_ui,
            outputs=[monitor_status, memory_display],
        ).then(
            fn=update_live_status,
            inputs=None,
            outputs=[live_status],
        )
        stop_guardian_btn.click(
            fn=stop_guardian_ui,
            outputs=[monitor_status, memory_display],
        ).then(
            fn=update_live_status,
            inputs=None,
            outputs=[live_status],
        )
        apply_profile_btn.click(
            fn=set_profile_ui,
            inputs=profile_dropdown,
            outputs=memory_display,
        ).then(
            fn=update_live_status,
            inputs=None,
            outputs=[live_status],
        )
        apply_thresholds_btn.click(
            fn=set_thresholds_ui,
            inputs=[low_slider, med_slider, high_slider, crit_slider],
            outputs=memory_display,
        ).then(
            fn=update_live_status,
            inputs=None,
            outputs=[live_status],
        )

        refresh_timer.tick(
            fn=lambda: (
                get_memory_stats_wrapper(state),
                get_monitor_status(),
            ),
            outputs=[memory_display, monitor_status],
        )


        # Initialize model selector and templates on load
        # Skip model initialization if in quick-start mode
        def initialize_ui():
            """Initialize UI components, respecting startup flags."""
            import sys
            
            # Check if --quick-start was used
            quick_start_mode = "--quick-start" in sys.argv
            
            if quick_start_mode:
                # In quick-start mode, don't auto-load models
                return (
                    gr.update(choices=[("⚡ Quick Start Mode - Models Not Loaded", "")], value=""),
                    "⚡ Quick Start Mode: Models can be loaded manually from the System Info tab",
                    refresh_project_list(),
                    refresh_template_list(),
                    *get_template_statistics(),
                    *refresh_gallery(),
                    gallery_filter.get("keyword", ""),
                    get_memory_stats_markdown(state),
                    get_monitor_status()
                )
            else:
                # Normal mode - refresh model list
                return (
                    refresh_model_list(),
                    update_model_info(CONFIG.sd_model),
                    refresh_project_list(),
                    refresh_template_list(),
                    *get_template_statistics(),
                    *refresh_gallery(),
                    gallery_filter.get("keyword", ""),
                    get_memory_stats_markdown(state),
                    get_monitor_status()
                )
        
        demo.load(
            fn=initialize_ui,
            outputs=[model_selector, model_info, project_selector, template_list, template_stats, popular_templates, gallery_component, tag_filter, page_display, keyword_filter, memory_display, monitor_status]
        )

        demo.load(
            fn=open_modal_fn,
            outputs=[first_run_modal]
        )
        
        # Component change handlers to track values
        prompt.change(fn=update_prompt_value, inputs=[prompt], outputs=[])
        negative_prompt.change(fn=update_negative_prompt_value, inputs=[negative_prompt], outputs=[])
        steps.change(fn=update_steps_value, inputs=[steps], outputs=[])
        guidance.change(fn=update_guidance_value, inputs=[guidance], outputs=[])
        seed.change(fn=update_seed_value, inputs=[seed], outputs=[])
        save_gallery.change(fn=update_save_gallery_value, inputs=[save_gallery], outputs=[])
        resolution.change(fn=update_resolution_value, inputs=[resolution], outputs=[])
        auto_best.change(fn=update_auto_best_value, inputs=[auto_best], outputs=[])
        
        # Quick Style Button Handlers
        def apply_style_prefix(current_prompt, style_prefix):
            """Apply a style prefix to the current prompt."""
            if not current_prompt:
                return style_prefix
            
            # Check if style is already applied
            if current_prompt.startswith(style_prefix):
                return current_prompt
            
            return f"{style_prefix}, {current_prompt}"
        
        anime_btn.click(
            fn=lambda p: apply_style_prefix(p, "anime style, detailed anime art"),
            inputs=[prompt],
            outputs=[prompt]
        )
        
        realistic_btn.click(
            fn=lambda p: apply_style_prefix(p, "photorealistic, high quality photography"),
            inputs=[prompt],
            outputs=[prompt]
        )
        
        artistic_btn.click(
            fn=lambda p: apply_style_prefix(p, "artistic masterpiece, fine art style"),
            inputs=[prompt],
            outputs=[prompt]
        )
        
        fantasy_btn.click(
            fn=lambda p: apply_style_prefix(p, "fantasy art, magical atmosphere"),
            inputs=[prompt],
            outputs=[prompt]
        )
        
        cyberpunk_btn.click(
            fn=lambda p: apply_style_prefix(p, "cyberpunk style, neon lights, futuristic"),
            inputs=[prompt],
            outputs=[prompt]
        )
        
    return demo

def parse_resolution(resolution_string):
    """Convert a resolution dropdown value to ``(width, height)``.

    The UI dropdown values are strings like ``"1024x1024 (Square - High Quality)"``.
    This helper extracts the numeric width and height components and returns
    sensible defaults if the format is unexpected.
    """

    if not isinstance(resolution_string, str):
        logger.warning(f"Invalid resolution value: {resolution_string}, using default 1024x1024")
        return 1024, 1024

    try:
        # Handle None, empty string, or whitespace-only strings
        if not resolution_string or not resolution_string.strip():
            logger.warning("Empty or None resolution string, using default 1024x1024")
            return 1024, 1024
            
        parts = resolution_string.strip().split()
        if not parts or len(parts) == 0:
            logger.warning("No parts found in resolution string, using default 1024x1024")
            return 1024, 1024
            
        # Safely access the first part
        resolution_part = parts[0] if len(parts) > 0 else ""
        if not resolution_part or "x" not in resolution_part:
            logger.warning(f"Invalid resolution format: {resolution_string}, using default 1024x1024")
            return 1024, 1024
            
        # Split width and height with additional validation
        resolution_components = resolution_part.split("x")
        if len(resolution_components) != 2:
            logger.warning(f"Invalid resolution components: {resolution_part}, using default 1024x1024")
            return 1024, 1024
            
        # Additional safety check - this should be redundant but ensures robustness
        if len(resolution_components) < 2:
            logger.warning(f"Insufficient resolution components after split: {resolution_components}, using default 1024x1024")
            return 1024, 1024
            
        width_str, height_str = resolution_components[0].strip(), resolution_components[1].strip()
        
        # Validate that we have non-empty strings
        if not width_str or not height_str:
            logger.warning(f"Empty width or height in resolution: {resolution_part}, using default 1024x1024")
            return 1024, 1024
            
        width, height = int(width_str), int(height_str)
        
        if width <= 0 or height <= 0:
            logger.warning(f"Invalid dimensions {width}x{height}, using default 1024x1024")
            return 1024, 1024
            
        # Clamp to reasonable ranges
        width = max(256, min(width, 2048))
        height = max(256, min(height, 2048))
        
        return width, height
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        logger.error(f"Error parsing resolution '{resolution_string}': {e}, using default 1024x1024")
        return 1024, 1024
    except Exception as e:
        logger.error(f"Unexpected error parsing resolution '{resolution_string}': {e}, using default 1024x1024")
        return 1024, 1024


def get_resolution_option(width, height):
    """Return the dropdown option string that matches the given dimensions."""
    for opt in RESOLUTION_OPTIONS:
        w, h = parse_resolution(opt)
        if w == width and h == height:
            return opt
    return "1024x1024 (Square - High Quality)"


def rename_project(state: AppState, old_name: str | None, new_name: str):
    """Rename a project directory."""
    new_name = (new_name or "").strip()
    if not old_name:
        return gr.update(), "Please select a project to rename"
    if not new_name:
        return gr.update(), "Please enter a new name"
    if old_name == new_name:
        return gr.update(), "Names are identical"
    old_path = PROJECTS_DIR / old_name
    new_path = PROJECTS_DIR / new_name
    if not old_path.exists():
        return gr.update(), "Project not found"
    if new_path.exists():
        return gr.update(), "A project with that name already exists"
    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))
    except Exception as e:  # pragma: no cover - unexpected errors
        logger.error("Failed to rename project %s to %s: %s", old_name, new_name, e)
        return gr.update(), f"Failed to rename project: {e}"
    if state.current_project == old_name:
        state.current_project = new_name
        current_value = new_name
    else:
        current_value = state.current_project
    choices = [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]
    return gr.update(choices=choices, value=current_value), f"Project '{old_name}' renamed to '{new_name}'"
