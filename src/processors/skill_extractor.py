"""
Skill extraction processor for Stage 3 enrichment pipeline.

Matches job descriptions against 73 skills defined in
``config/skill_taxonomy.yaml``. Each skill has pre-compiled regex
patterns applied case-insensitively. Results are grouped by category
and also provided as a flat list.
"""

import re
from typing import Any

import yaml

from src.models import RawJob
from src.processors.base import BaseProcessor
from src.utils.Io import PROJECT_ROOT


class SkillExtractor(BaseProcessor):
    """
    Extract skills from job descriptions using regex pattern matching.

    Loads the skill taxonomy once at init and pre-compiles all patterns
    for efficient batch processing.
    """

    def __init__(self) -> None:
        super().__init__()
        config_path = PROJECT_ROOT / "config" / "skill_taxonomy.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            taxonomy = yaml.safe_load(fh)

        # Build lookup: {skill_name: (category, [compiled_patterns])}
        self._skills: dict[str, tuple[str, list[re.Pattern]]] = {}
        for skill_name, spec in taxonomy.items():
            category = spec["category"]
            patterns = [
                re.compile(p, re.IGNORECASE) for p in spec["patterns"]
            ]
            self._skills[skill_name] = (category, patterns)

        self.logger.info(
            "SkillExtractor initialized: %d skills across categories",
            len(self._skills),
        )

    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """
        Match skills against the job description.

        Args:
            job: Raw job object (title is also searched).
            text: Cleaned description text.

        Returns:
            Dict with ``skills``, ``all_skills_flat``, ``skill_count``.
        """
        # Combine title and description for matching
        search_text = f"{job.title} {text}"

        skills_by_category: dict[str, list[str]] = {}
        flat: list[str] = []

        for skill_name, (category, patterns) in self._skills.items():
            for pattern in patterns:
                if pattern.search(search_text):
                    skills_by_category.setdefault(category, []).append(skill_name)
                    flat.append(skill_name)
                    break  # One match per skill is enough

        return {
            "skills": skills_by_category,
            "all_skills_flat": flat,
            "skill_count": len(flat),
        }
