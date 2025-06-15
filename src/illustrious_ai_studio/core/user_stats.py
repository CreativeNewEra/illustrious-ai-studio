"""User statistics tracking utilities."""

import json
import logging
from pathlib import Path
from datetime import datetime, date

logger = logging.getLogger(__name__)


class UserStats:
    """Track user creation statistics."""

    def __init__(self) -> None:
        self.stats_file = Path("user_data/stats.json")
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        self.load_stats()

    def load_stats(self) -> None:
        """Load user statistics."""
        if self.stats_file.exists():
            with open(self.stats_file) as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(
                        "The stats file is corrupted. " "Reinitializing data."
                    )
                    self.data = {
                        "total_images": 0,
                        "total_chats": 0,
                        "favorite_styles": {},
                        "daily_creations": {},
                        "achievements": [],
                        "creation_times": [],
                    }
        else:
            self.data = {
                "total_images": 0,
                "total_chats": 0,
                "favorite_styles": {},
                "daily_creations": {},
                "creation_times": [],
            }

    def track_creation(self, style: str = "default") -> None:
        """Track a new image creation."""
        self.data["total_images"] += 1

        self.data["favorite_styles"][style] = (
            self.data["favorite_styles"].get(style, 0) + 1
        )

        today = date.today().isoformat()
        self.data["daily_creations"][today] = (
            self.data["daily_creations"].get(today, 0) + 1
        )

        self.data["creation_times"].append(datetime.now().isoformat())

        self.save_stats()

    def get_streak(self) -> int:
        """Calculate current daily streak."""
        if not self.data["daily_creations"]:
            return 0

        dates = sorted(self.data["daily_creations"].keys(), reverse=True)
        streak = 0
        current_date = date.today()

        for date_str in dates:
            check_date = date.fromisoformat(date_str)
            if (current_date - check_date).days == streak:
                streak += 1
            else:
                break

        return streak

    def get_stats_display(self) -> str:
        """Get formatted stats for display."""
        total = self.data["total_images"]
        # fmt: off
        favorites = len(
            [k for k, v in self.data["favorite_styles"].items() if v > 2]
        )
        # fmt: on
        streak = self.get_streak()

        return (
            "### 📊 Your Creative Stats\n"
            f"🎨 Images Created: {total}\n"
            f"⭐ Favorite Styles: {favorites}\n"
            f"🎯 Current Streak: {streak} days\n"
            f"🏆 Next Achievement: {self.get_next_achievement()}"
        )

    def get_next_achievement(self) -> str:
        """Get the next achievement to work towards."""
        total = self.data["total_images"]

        milestones = [
            (1, "First Creation"),
            (5, "Getting Started"),
            (10, "Creative Explorer"),
            (25, "Art Enthusiast"),
            (50, "Prolific Creator"),
            (100, "Master Artist"),
        ]

        for count, title in milestones:
            if total < count:
                return f"{title} ({total}/{count})"

        return "Legendary Creator!"

    def save_stats(self) -> None:
        """Save statistics to file."""
        with open(self.stats_file, "w") as f:
            json.dump(self.data, f, indent=2)
