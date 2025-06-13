import json
import logging
import uuid
from pathlib import Path
import os
import subprocess
import sys

import gradio as gr

from core.sdxl import generate_image, TEMP_DIR, get_latest_image, init_sdxl, get_available_models, get_current_model_info, test_model_generation, switch_sdxl_model
from core.config import CONFIG
from core.ollama import generate_prompt, handle_chat, analyze_image, init_ollama
from core import sdxl, ollama
from core.memory import get_model_status, get_memory_stats_markdown, get_memory_stats_wrapper
from core.memory_guardian import (
    start_memory_guardian,
    stop_memory_guardian,
    get_memory_guardian,
)
from core.state import AppState
from core.prompt_templates import template_manager

logger = logging.getLogger(__name__)

THEME_PREF_FILE = TEMP_DIR / "theme_pref.json"

def create_gradio_app(state: AppState):
    """Build and return the Gradio UI for the application."""
    css_file = (Path(__file__).parent / "custom.css").read_text()
    def load_theme_pref():
        try:
            if THEME_PREF_FILE.exists():
                with open(THEME_PREF_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("theme", "light")
        except Exception as e:
            logger.error("Error loading theme preference: %s", e)
        return "light"

    def save_theme_pref(choice: str):
        try:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            with open(THEME_PREF_FILE, "w", encoding="utf-8") as f:
                json.dump({"theme": choice}, f)
        except Exception as e:
            logger.error("Error saving theme preference: %s", e)
        return choice

    gallery_files: list[Path] = []

    def load_gallery_items():
        """Return gallery images with captions and track file paths."""
        nonlocal gallery_files
        gallery_dir = Path(CONFIG.gallery_dir)
        gallery_dir.mkdir(parents=True, exist_ok=True)
        items: list[tuple[str, str]] = []
        gallery_files = []
        for img_path in sorted(gallery_dir.glob("*.png"), reverse=True):
            caption = ""
            meta_path = img_path.with_suffix(".json")
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    caption = meta.get("prompt", "")
                except Exception as e:  # pragma: no cover - non-critical
                    logger.error("Failed to load metadata for %s: %s", img_path, e)
            items.append((str(img_path), caption))
            gallery_files.append(img_path)
        return items

    def refresh_gallery():
        """Update gallery component value."""
        return gr.update(value=load_gallery_items())

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

    current_theme = load_theme_pref()
    with gr.Blocks(
        title="Illustrious AI Studio",
        theme="default",
        css=css_file
    ) as demo:
        with gr.Row(elem_classes=["main-container"]):
            with gr.Column():
                gr.Markdown("""
                    # 🎨 Illustrious AI Studio
                    ### Your Creative AI Workspace
                    Transform your imagination into stunning artwork with the power of AI
                """)
                status_display = gr.Markdown(get_model_status(state))
                theme_selector = gr.Radio(
                    ["Light", "Dark"],
                    value="Dark" if current_theme == "dark" else "Light",
                    label="Theme",
                    interactive=True,
                )
        with gr.Tab("🎨 Text-to-Image"):
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
                    prompt = gr.Textbox(label="Prompt", placeholder="Describe what you want to create...", lines=3)
                    
                    # Quick Style Buttons
                    with gr.Row():
                        gr.Markdown("**🎨 Quick Styles:**")
                    with gr.Row():
                        anime_btn = gr.Button("🌸 Anime", variant="secondary", size="sm")
                        realistic_btn = gr.Button("📷 Realistic", variant="secondary", size="sm")
                        artistic_btn = gr.Button("🎭 Artistic", variant="secondary", size="sm")
                        fantasy_btn = gr.Button("🧙 Fantasy", variant="secondary", size="sm")
                        cyberpunk_btn = gr.Button("🤖 Cyberpunk", variant="secondary", size="sm")
                    
                    with gr.Accordion("🎯 Creative Controls", open=False):
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
                        
                        negative_prompt = gr.Textbox(
                            label="Elements to Avoid",
                            value="blurry, low quality, text, watermark, deformed",
                            lines=2,
                            elem_classes=["textbox"]
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
                                choices=[
                                    "512x512 (Square - Fast)",
                                    "768x768 (Square - Balanced)",
                                    "1024x1024 (Square - High Quality)",
                                    "768x512 (Landscape)",
                                    "512x768 (Portrait)",
                                    "1024x768 (Landscape HD)",
                                    "768x1024 (Portrait HD)"
                                ],
                                value="1024x1024 (Square - High Quality)",
                                elem_classes=["dropdown"]
                            )
                        seed = gr.Number(
                            value=-1,
                            label="Inspiration Seed (-1 for random)",
                            elem_classes=["number-input"]
                        )
                        save_gallery = gr.Checkbox(
                            value=True,
                            label="Add to Gallery Collection",
                            elem_classes=["checkbox-input"]
                        )
                    with gr.Row():
                        generate_btn = gr.Button(
                            "🎨 Create Masterpiece",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-button"]
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
        with gr.Tab("💬 AI Chat & Prompt Crafting"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        height=500,
                        show_copy_button=True,
                        elem_classes=["chatbot"]
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
        with gr.Tab("🔍 Image Analysis"):
            if state.model_status["multimodal"]:
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
            else:
                gr.Markdown("## ❌ Multimodal Analysis Unavailable")
                gr.Markdown("Please ensure you have a multimodal LLM and mmproj model configured.")
                
        with gr.Tab("📝 Prompt Templates"):
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

        with gr.Tab("🖼️ Gallery"):
            gallery_component = gr.Gallery(
                label="Gallery",
                elem_classes=["gallery-section"],
                columns=4,
                height="auto",
            )
            metadata_display = gr.JSON(label="Metadata")
            selected_path = gr.Textbox(visible=False)
            with gr.Row():
                open_file_btn = gr.Button(
                    "📂 Open File",
                    variant="secondary",
                    elem_classes=["secondary-button"],
                )
                copy_path_btn = gr.Button(
                    "📋 Copy Path",
                    variant="secondary",
                    elem_classes=["secondary-button"],
                )
            action_status = gr.Textbox(
                label="Action Status",
                interactive=False,
                elem_classes=["status-box"],
            )

        with gr.Tab("📊 System Info"):
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
                return gr.update(choices=[("❌ Error loading models", "")], value="")
        
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
                return f"❌ Error: {str(e)}"
        
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
                    status_msg = "❌ Failed to switch model - check logs for details"
                
                return (
                    gr.update(visible=True, value=status_msg),
                    update_model_info(selected_path),
                    get_model_status(state),
                    get_memory_stats_markdown(state)
                )
                
            except Exception as e:
                logger.error("Model switch failed: %s", e)
                return (
                    gr.update(visible=True, value=f"❌ Switch failed: {str(e)}"),
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
                        gr.update(visible=True, value=f"❌ Test failed: {message}"),
                        gr.update()
                    )
                    
            except Exception as e:
                logger.error("Model test failed: %s", e)
                return (
                    gr.update(visible=True, value=f"❌ Test error: {str(e)}"),
                    gr.update()
                )
        
        # Event Handlers
        theme_selector.change(
            fn=lambda m: save_theme_pref(m.lower()),
            inputs=theme_selector,
            outputs=[],
            js="(mode) => { if(mode === 'Dark'){ document.documentElement.classList.add('dark'); } else { document.documentElement.classList.remove('dark'); } }"
        )
        enhance_btn.click(fn=lambda p: generate_prompt(state, p), inputs=prompt, outputs=prompt)
        
        # Updated generate button to use wrapper and update recent prompts
        def generate_and_update_history(p, n, st, g, se, save_flag, res):
            # Parse resolution
            width, height = parse_resolution(res)
            
            # Generate the image with resolution
            image, status = generate_image(state, p, n, st, g, se, save_flag, width, height)
            
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
        
        # Regenerate function
        def regenerate_image():
            if state.last_generation_params is None:
                return None, "❌ No previous generation to repeat", gr.update()
            
            params = state.last_generation_params
            image, status = generate_image(
                state, 
                params["prompt"], 
                params["negative_prompt"], 
                params["steps"], 
                params["guidance"], 
                params["seed"], 
                params["save_gallery"], 
                params["width"], 
                params["height"]
            )
            
            if image is not None and params["prompt"].strip():
                updated_prompts = add_to_recent_prompts(params["prompt"].strip())
                return image, status, gr.update(choices=updated_prompts)
            else:
                return image, status, gr.update()
        
        generate_btn.click(
            fn=generate_and_update_history,
            inputs=[prompt, negative_prompt, steps, guidance, seed, save_gallery, resolution],
            outputs=[output_image, generation_status, recent_prompts, regenerate_btn],
        ).then(
            fn=refresh_gallery,
            inputs=None,
            outputs=gallery_component,
        )

        regenerate_btn.click(
            fn=regenerate_image,
            inputs=[],
            outputs=[output_image, generation_status, recent_prompts],
        ).then(
            fn=refresh_gallery,
            inputs=None,
            outputs=gallery_component,
        )
        
        # Recent prompts selection handler
        recent_prompts.change(
            fn=select_recent_prompt,
            inputs=[recent_prompts],
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
            js=f"() => {{ if('{current_theme}' === 'dark') {{ document.documentElement.classList.add('dark'); }} else {{ document.documentElement.classList.remove('dark'); }} }}"
        )

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

        send_btn.click(fn=chat_wrapper_with_image_update, inputs=[msg, chatbot], outputs=[chatbot, msg, output_image])
        msg.submit(fn=chat_wrapper_with_image_update, inputs=[msg, chatbot], outputs=[chatbot, msg, output_image])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        if state.model_status["multimodal"]:
            analyze_btn.click(fn=lambda img, q: analyze_image(state, img, q), inputs=[input_image, analysis_question], outputs=analysis_output)
        
        # Model Selection Event Handlers
        model_selector.change(
            fn=update_model_info,
            inputs=model_selector,
            outputs=model_info
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
                return f"❌ Failed to save template: {str(e)}"
        
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
                return "❌ Failed to delete template"
        
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
                return f"❌ Import failed: {str(e)}"
        
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
            fn=lambda: refresh_template_list(),
            outputs=template_list
        )
        
        export_btn.click(
            fn=export_templates,
            outputs=export_download
        )
        
        import_file.change(
            fn=import_templates,
            inputs=[import_file, import_merge],
            outputs=template_save_status
        ).then(
            fn=lambda: refresh_template_list(),
            outputs=template_list
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

        def get_monitor_status():
            guardian = get_memory_guardian(state)
            return "🟢 Monitoring" if guardian.is_monitoring else "🔴 Stopped"

        def start_guardian_ui():
            start_memory_guardian(state)
            return get_monitor_status(), get_memory_stats_markdown(state)

        def stop_guardian_ui():
            stop_memory_guardian()
            return get_monitor_status(), get_memory_stats_markdown(state)
        
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
        switch_btn.click(
            fn=do_switch,
            inputs=[sd_model_input, ollama_model_input],
            outputs=[status_display, config_display, memory_display, monitor_status],
        )
        refresh_btn.click(
            fn=lambda: (
                get_model_status(state),
                get_memory_stats_markdown(state),
                get_monitor_status(),
            ),
            outputs=[status_display, memory_display, monitor_status],
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

        load_selected_btn.click(
            fn=load_selected_models,
            inputs=[sdxl_checkbox, ollama_checkbox, vision_checkbox],
            outputs=[status_display, memory_display, monitor_status],
        )

        start_guardian_btn.click(
            fn=start_guardian_ui,
            outputs=[monitor_status, memory_display],
        )
        stop_guardian_btn.click(
            fn=stop_guardian_ui,
            outputs=[monitor_status, memory_display],
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
                    refresh_template_list(),
                    *get_template_statistics(),
                    refresh_gallery(),
                    get_memory_stats_markdown(state),
                    get_monitor_status()
                )
            else:
                # Normal mode - refresh model list
                return (
                    refresh_model_list(),
                    update_model_info(CONFIG.sd_model),
                    refresh_template_list(),
                    *get_template_statistics(),
                    refresh_gallery(),
                    get_memory_stats_markdown(state),
                    get_monitor_status()
                )
        
        demo.load(
            fn=initialize_ui,
            outputs=[model_selector, model_info, template_list, template_stats, popular_templates, gallery_component, memory_display, monitor_status]
        )
        
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
    """Parse resolution string to width and height."""
    try:
        # Extract resolution from strings like "1024x1024 (Square - High Quality)"
        resolution_part = resolution_string.split(' ')[0]  # Get "1024x1024"
        width, height = map(int, resolution_part.split('x'))
        return width, height
    except (ValueError, IndexError):
        # Default to 1024x1024 if parsing fails
        return 1024, 1024
