"""
Reports package — Stage 8.

Renders an ``AnalyticsResult`` into a markdown executive report
suitable for sharing with hiring managers, compliance teams, or
recruiters.
"""

from src.reports.report_generator import ReportGenerator, render_report

__all__ = ["ReportGenerator", "render_report"]
