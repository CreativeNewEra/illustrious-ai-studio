"""
Prompt Template Management System
Allows users to save, load, export, and import prompt templates.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class PromptTemplateManager:
    """Manages prompt templates for the AI Studio."""
    
    def __init__(self, templates_dir: Path = None):
        self.templates_dir = templates_dir or Path("user_prompts")
        self.templates_dir.mkdir(exist_ok=True)
        self.templates_file = self.templates_dir / "user_templates.json"
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from the JSON file."""
        if self.templates_file.exists():
            try:
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load templates: {e}")
                return {"templates": [], "categories": ["General", "Character", "Scene", "Style"]}
        else:
            # Create default structure
            return {
                "templates": [],
                "categories": ["General", "Character", "Scene", "Style"],
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
    
    def _save_templates(self) -> bool:
        """Save templates to the JSON file."""
        try:
            self.templates["updated_at"] = datetime.now().isoformat()
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")
            return False
    
    def add_template(self, name: str, prompt: str, negative_prompt: str = "", 
                    category: str = "General", tags: List[str] = None,
                    settings: Dict[str, Any] = None) -> str:
        """Add a new template."""
        template_id = str(uuid.uuid4())
        template = {
            "id": template_id,
            "name": name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "category": category,
            "tags": tags or [],
            "settings": settings or {},
            "created_at": datetime.now().isoformat(),
            "usage_count": 0
        }
        
        self.templates["templates"].append(template)
        if self._save_templates():
            logger.info(f"Template '{name}' saved successfully")
            return template_id
        else:
            raise Exception("Failed to save template")
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a template by ID."""
        for template in self.templates["templates"]:
            if template["id"] == template_id:
                return template
        return None
    
    def get_templates_by_category(self, category: str = None) -> List[Dict[str, Any]]:
        """Get templates by category, or all if no category specified."""
        if category is None:
            return self.templates["templates"]
        return [t for t in self.templates["templates"] if t["category"] == category]
    
    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """Search templates by name, prompt, or tags."""
        query = query.lower()
        results = []
        for template in self.templates["templates"]:
            if (query in template["name"].lower() or 
                query in template["prompt"].lower() or
                any(query in tag.lower() for tag in template["tags"])):
                results.append(template)
        return results
    
    def update_template(self, template_id: str, **kwargs) -> bool:
        """Update an existing template."""
        for i, template in enumerate(self.templates["templates"]):
            if template["id"] == template_id:
                for key, value in kwargs.items():
                    if key in template:
                        template[key] = value
                template["updated_at"] = datetime.now().isoformat()
                return self._save_templates()
        return False
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        original_length = len(self.templates["templates"])
        self.templates["templates"] = [
            t for t in self.templates["templates"] if t["id"] != template_id
        ]
        if len(self.templates["templates"]) < original_length:
            return self._save_templates()
        return False
    
    def increment_usage(self, template_id: str) -> bool:
        """Increment usage count for a template."""
        for template in self.templates["templates"]:
            if template["id"] == template_id:
                template["usage_count"] = template.get("usage_count", 0) + 1
                template["last_used"] = datetime.now().isoformat()
                return self._save_templates()
        return False
    
    def get_popular_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular templates by usage count."""
        sorted_templates = sorted(
            self.templates["templates"],
            key=lambda x: x.get("usage_count", 0),
            reverse=True
        )
        return sorted_templates[:limit]
    
    def export_templates(self, export_path: Path = None) -> str:
        """Export templates to a JSON file."""
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.templates_dir / f"exported_templates_{timestamp}.json"
        
        export_data = {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "total_templates": len(self.templates["templates"]),
                "version": self.templates.get("version", "1.0")
            },
            "templates": self.templates["templates"],
            "categories": self.templates["categories"]
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Templates exported to {export_path}")
            return str(export_path)
        except Exception as e:
            logger.error(f"Failed to export templates: {e}")
            raise
    
    def import_templates(self, import_path: Path, merge: bool = True) -> int:
        """Import templates from a JSON file."""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_templates = import_data.get("templates", [])
            imported_categories = import_data.get("categories", [])
            
            if not merge:
                # Replace all templates
                self.templates["templates"] = imported_templates
                self.templates["categories"] = imported_categories
            else:
                # Merge templates (avoid duplicates by name)
                existing_names = {t["name"] for t in self.templates["templates"]}
                new_templates = []
                
                for template in imported_templates:
                    if template["name"] not in existing_names:
                        # Generate new ID for imported template
                        template["id"] = str(uuid.uuid4())
                        template["imported_at"] = datetime.now().isoformat()
                        new_templates.append(template)
                
                self.templates["templates"].extend(new_templates)
                
                # Merge categories
                for category in imported_categories:
                    if category not in self.templates["categories"]:
                        self.templates["categories"].append(category)
            
            if self._save_templates():
                count = len(imported_templates) if not merge else len(new_templates)
                logger.info(f"Successfully imported {count} templates")
                return count
            else:
                raise Exception("Failed to save imported templates")
                
        except Exception as e:
            logger.error(f"Failed to import templates: {e}")
            raise
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about the template collection."""
        templates = self.templates["templates"]
        categories = {}
        total_usage = 0
        
        for template in templates:
            category = template["category"]
            categories[category] = categories.get(category, 0) + 1
            total_usage += template.get("usage_count", 0)
        
        return {
            "total_templates": len(templates),
            "categories": categories,
            "total_usage": total_usage,
            "average_usage": total_usage / len(templates) if templates else 0,
            "most_popular": self.get_popular_templates(1)[0] if templates else None
        }

# Global instance
template_manager = PromptTemplateManager()
