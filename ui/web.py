import json
import logging
import uuid
from pathlib import Path

import gradio as gr

from core.sdxl import generate_image, TEMP_DIR, get_latest_image
from core.config import CONFIG
from core.ollama import generate_prompt, handle_chat, analyze_image
from core.memory import get_model_status
from core.state import AppState
from core.prompt_templates import template_manager

logger = logging.getLogger(__name__)


def create_gradio_app(state: AppState):
    """Build and return the Gradio UI for the application."""
    css_file = (Path(__file__).parent / "custom.css").read_text()
    with gr.Blocks(
        title="Illustrious AI Studio",
        theme=gr.themes.Base(),
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
        with gr.Tab("🎨 Text-to-Image"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", placeholder="Describe what you want to create...", lines=3)
                    with gr.Accordion("🎯 Creative Controls", open=False):
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
                            label="Upload Artwork",
                            type="pil",
                            elem_classes=["gallery-item"]
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
        enhance_btn.click(fn=lambda p: generate_prompt(state, p), inputs=prompt, outputs=prompt)
        generate_btn.click(
            fn=lambda p, n, st, g, se, save_flag: generate_image(state, p, n, st, g, se, save_flag),
            inputs=[prompt, negative_prompt, steps, guidance, seed, save_gallery],
            outputs=[output_image, generation_status],
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
            return result_history, empty_msg, gr.update()

        send_btn.click(fn=chat_wrapper_with_image_update, inputs=[msg, chatbot], outputs=[chatbot, msg, output_image])
        msg.submit(fn=chat_wrapper_with_image_update, inputs=[msg, chatbot], outputs=[chatbot, msg, output_image])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        if state.model_status["multimodal"]:
            analyze_btn.click(fn=lambda img, q: analyze_image(state, img, q), inputs=[input_image, analysis_question], outputs=analysis_output)
        
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
        
        # Initialize template list and stats on load
        demo.load(
            fn=lambda: (refresh_template_list(), *get_template_statistics()),
            outputs=[template_list, template_stats, popular_templates]
        )
        
        def do_switch(sd_path, ollama_name):
            if sd_path:
                sdxl.switch_sdxl_model(state, sd_path)
            if ollama_name:
                ollama.switch_ollama_model(state, ollama_name)
            return get_model_status(state), json.dumps(CONFIG.as_dict(), indent=2)
        switch_btn.click(fn=do_switch, inputs=[sd_model_input, ollama_model_input], outputs=[status_display, config_display])
        refresh_btn.click(fn=lambda: get_model_status(state), outputs=status_display)
    return demo
