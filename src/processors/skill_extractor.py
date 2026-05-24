"""
Skill extractor — matches job text against the 73-skill taxonomy.

Loads ``config/skill_taxonomy.yaml`` once at init and pre-compiles
every pattern. Each skill has one or more regex patterns; one match
on any pattern attributes that skill to the job. Results are returned
both grouped by category (for dashboard breakdowns) and as a flat
list (for embedding metadata and downstream filters).
"""

import re
from typing import Any

import yaml

from src.models import RawJob
from src.processors.base import BaseProcessor
from src.utils.io import PROJECT_ROOT


class SkillExtractor(BaseProcessor):
    """Extract skills from job text via regex pattern matching."""

    def __init__(self) -> None:
        super().__init__()
        config_path = PROJECT_ROOT / "config" / "skill_taxonomy.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            taxonomy = yaml.safe_load(fh)

        # {skill_name: (category, [compiled_patterns])}
        self._skills: dict[str, tuple[str, list[re.Pattern]]] = {}
        for skill_name, spec in taxonomy.items():
            category = spec["category"]
            patterns = [
                re.compile(p, re.IGNORECASE) for p in spec["patterns"]
            ]
            self._skills[skill_name] = (category, patterns)

        self.logger.info(
            "SkillExtractor initialized: %d skills loaded", len(self._skills),
        )

    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """
        Match every skill in the taxonomy against title + description.

        Returns ``skills`` (grouped by category), ``all_skills_flat``
        (flat list), and ``skill_count``.
        """
        search_text = f"{job.title} {text}"

        skills_by_category: dict[str, list[str]] = {}
        flat: list[str] = []

        for skill_name, (category, patterns) in self._skills.items():
            for pattern in patterns:
                if pattern.search(search_text):
                    skills_by_category.setdefault(category, []).append(skill_name)
                    flat.append(skill_name)
                    break  # one match per skill is enough

        return {
            "skills": skills_by_category,
            "all_skills_flat": flat,
            "skill_count": len(flat),
        }
