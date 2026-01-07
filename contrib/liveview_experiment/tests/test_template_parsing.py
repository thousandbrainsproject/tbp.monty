"""Unit tests for template parsing - catch template errors before server startup."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pyview.template.live_template import LiveTemplate
from pyview.vendor import ibis
from pyview.vendor.ibis.loaders import FileReloader

from contrib.liveview_experiment.src.visualization_state import (
    EvidencePoint,
    SensorImages,
    VisualizationState,
)


class TestTemplateParsing:
    """Test that the experiment.html template can be parsed and rendered."""

    def _load_template(self) -> ibis.Template:
        """Load the experiment.html template using the same method as LiveView.
        
        Returns the underlying ibis.Template directly for testing.
        """
        current_file_path = Path(__file__).resolve()
        templates_dir = current_file_path.parent.parent / "templates"

        if not hasattr(ibis, "loader") or not isinstance(ibis.loader, FileReloader):
            ibis.loader = FileReloader(str(templates_dir))

        template_path = "experiment.html"
        template_file = templates_dir / template_path
        template_content = template_file.read_text(encoding="utf-8")

        template = ibis.Template(template_content)
        return template

    def _create_minimal_template_assigns(self) -> dict:
        """Create minimal template assigns for testing."""
        # Create a minimal evidence history with one point
        evidence_point = EvidencePoint(
            step=0,
            evidences={"object1": 0.5, "object2": 0.3},
            target_object="object1",
            episode=0,
        )
        evidence_history = [evidence_point]

        # Create minimal visualization state
        from collections import deque
        from markupsafe import Markup
        viz_state = VisualizationState(
            evidence_history=deque(evidence_history),
            sensor_images=SensorImages(),
        )

        # Minimal template assigns matching what _build_template_assigns produces
        return {
            # String assigns
            "run_name": "test_run",
            "experiment_name": "test_experiment",
            "environment_name": "test_env",
            "config_path": "/path/to/config.yaml",
            # Status and mode
            "status": "running",
            "status_display": "RUNNING",
            "experiment_mode": "train",
            "experiment_mode_display": "training",
            "experiment_mode_display_upper": "TRAINING",
            # Numeric assigns
            "total_train_steps": 100,
            "train_episodes": 10,
            "train_epochs": 5,
            "total_eval_steps": 50,
            "eval_episodes": 5,
            "eval_epochs": 2,
            "current_epoch": 1,
            "current_episode": 2,
            "current_step": 10,
            # Optional numeric assigns
            "max_train_steps": 1000,
            "max_eval_steps": 500,
            "max_total_steps": 1500,
            "n_train_epochs": 10,
            "n_eval_epochs": 5,
            # Boolean assigns
            "do_train": True,
            "do_eval": True,
            "show_sensor_output": False,
            # Time values
            "elapsed_time": "00:00:10",
            "last_update": "2024-01-01 12:00:00",
            # Complex values
            "data_streams": {},
            "recent_logs": [],
            # Visualization assigns
            "chart_data_json": Markup(viz_state.get_chart_data_json()),
            "chart_point_count": 1,
            "current_max_evidence_object": "object1",
            "current_max_evidence_value": "0.500",
            "camera_image_b64": None,
            "depth_image_b64": None,
            "sensor_image_step": 0,
            "has_sensor_images": False,
        }

    def test_template_can_be_loaded(self) -> None:
        """Test that the template file can be loaded without errors."""
        template = self._load_template()
        assert template is not None

    def test_template_can_be_rendered_with_minimal_data(self) -> None:
        """Test that the template can be rendered with minimal data."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        # Render the template - this will raise an exception if there are syntax errors
        result = template.render(**assigns)

        # Verify we got some HTML output
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify key elements are present
        assert "experiment.html" in result or "<html" in result.lower() or "<div" in result

    def test_template_renders_with_empty_evidence_history(self) -> None:
        """Test that the template renders correctly with empty evidence history."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        # Set empty evidence history in visualization state
        from collections import deque
        from contrib.liveview_experiment.src.visualization_state import VisualizationState
        from markupsafe import Markup
        empty_viz_state = VisualizationState(evidence_history=deque())
        assigns["chart_data_json"] = Markup(empty_viz_state.get_chart_data_json())
        assigns["chart_point_count"] = 0

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)

    def test_template_renders_with_multiple_evidence_points(self) -> None:
        """Test that the template renders correctly with multiple evidence points."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        # Create multiple evidence points
        points = [
            EvidencePoint(
                step=i,
                evidences={"object1": 0.5 + i * 0.1, "object2": 0.3 + i * 0.05},
                target_object="object1",
                episode=i // 10,
            )
            for i in range(5)
        ]
        from collections import deque
        evidence_history = deque(points)
        assigns["evidence_history"] = evidence_history
        assigns["chart_point_count"] = 5

        # Should render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)

        # Verify evidence points are in the output (as JSON in script tag)
        assert 'chart-data' in result or 'script type="application/json"' in result

    def test_template_handles_missing_optional_fields(self) -> None:
        """Test that the template handles missing optional fields gracefully."""
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()

        # Remove optional fields
        assigns.pop("max_train_steps", None)
        assigns.pop("max_eval_steps", None)
        assigns.pop("max_total_steps", None)
        assigns.pop("n_train_epochs", None)
        assigns.pop("n_eval_epochs", None)
        assigns.pop("environment_name", None)
        assigns.pop("experiment_name", None)
        assigns.pop("config_path", None)

        # Should still render without errors
        result = template.render(**assigns)
        assert result is not None
        assert isinstance(result, str)

    def test_evidence_point_to_dict_serialization(self) -> None:
        """Test that EvidencePoint.to_dict produces valid JSON-serializable data."""
        import json
        
        # Create a point with various data types
        point = EvidencePoint(
            step=42,
            evidences={"object1": 0.75, "object2": 0.25},
            target_object="object1",
            hypothesis_count=5,
            episode=10,
            timestamp=1234567890.123,
        )
        
        # Convert to dict and serialize to JSON
        point_dict = point.to_dict()
        json_str = json.dumps(point_dict)
        
        # Should be valid JSON
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # JSONL format requirement: no newlines (single-line JSON)
        assert '\n' not in json_str, "JSONL format must not contain newlines"
        assert '\r' not in json_str, "JSONL format must not contain carriage returns"
        
        # Parse directly
        parsed = json.loads(json_str)
        
        # Verify all fields are present and correct
        assert parsed["step"] == 42
        assert parsed["evidences"] == {"object1": 0.75, "object2": 0.25}
        assert parsed["target_object"] == "object1"
        assert parsed["hypothesis_count"] == 5
        assert parsed["episode"] == 10
        assert parsed["timestamp"] == 1234567890.123
        
        # Verify special characters are preserved (JSON-escaped, not HTML-escaped)
        point_with_special = EvidencePoint(
            step=1,
            evidences={"obj<test>": 0.5, "obj\"quote\"": 0.3},
            target_object="target&value",
        )
        json_str_special = json.dumps(point_with_special.to_dict())
        
        # Should be valid JSON (special chars are JSON-escaped, not HTML-escaped)
        parsed_special = json.loads(json_str_special)
        assert "obj<test>" in parsed_special["evidences"]
        assert "obj\"quote\"" in parsed_special["evidences"]
        assert parsed_special["target_object"] == "target&value"

    def test_chart_data_json_in_template_rendering(self) -> None:
        """Characterization test: Verify chart_data_json rendering in script tag."""
        import json
        
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()
        
        # Create evidence points with special characters
        points = [
            EvidencePoint(
                step=1,
                evidences={"obj<test>": 0.5, "obj\"quote\"": 0.3},
                target_object="target&value",
            ),
            EvidencePoint(
                step=2,
                evidences={"normal": 0.8},
                target_object="</pre><script>alert(1)</script><pre>",  # XSS attempt
            ),
        ]
        from collections import deque
        evidence_history = deque(points)
        
        # Update visualization state to include these points
        from contrib.liveview_experiment.src.visualization_state import VisualizationState
        from markupsafe import Markup
        viz_state = VisualizationState(evidence_history=evidence_history)
        assigns["chart_data_json"] = Markup(viz_state.get_chart_data_json())
        assigns["chart_point_count"] = len(points)
        
        # Render template
        result = template.render(**assigns)
        
        # Verify script tag is present
        assert 'script type="application/json"' in result or 'id="chart-data"' in result
        
        # Extract JSON from script tag
        import re
        import html
        
        # Find the script tag by ID and extract content
        # Need to handle case where JSON might contain </script> as a string
        # Find the opening tag
        script_start = result.find('<script')
        if script_start == -1:
            assert False, f"No script tag found. Result: {result[:500]}"
        
        # Find the id="chart-data" attribute
        chart_data_start = result.find('id="chart-data"', script_start)
        if chart_data_start == -1:
            assert False, f"No chart-data script tag found. Result: {result[:500]}"
        
        # Find the closing > of the opening tag
        tag_end = result.find('>', chart_data_start)
        if tag_end == -1:
            assert False, f"Script tag not properly closed. Result: {result[:500]}"
        
        # Find the actual closing </script> tag (not one inside JSON)
        # Start from the end and work backwards to find the last </script>
        script_end = result.rfind('</script>')
        if script_end == -1:
            assert False, f"No closing script tag found. Result: {result[:500]}"
        
        # Extract JSON content between tag_end+1 and script_end
        json_content = result[tag_end + 1:script_end].strip()
        
        # Unescape HTML entities (template engine may escape even with Markup)
        # Check if content is HTML-escaped
        if '&quot;' in json_content or '&#34;' in json_content:
            json_content = html.unescape(json_content)
            # If still has entities, unescape again (might be double-escaped)
            if '&quot;' in json_content or '&#34;' in json_content:
                json_content = html.unescape(json_content)
        
        # Verify we have valid JSON structure
        assert json_content.startswith('{'), f"JSON should start with {{, got: {json_content[:100]}"
        assert json_content.endswith('}'), f"JSON should end with }}, got: {json_content[-100:]}"
        
        parsed = json.loads(json_content)
        
        # Verify structure
        assert "evidence_history" in parsed
        assert len(parsed["evidence_history"]) == len(points)
        
        # Verify special characters are preserved in parsed data
        point1 = parsed["evidence_history"][0]
        assert "obj<test>" in point1["evidences"]
        assert "obj\"quote\"" in point1["evidences"]
        assert point1["target_object"] == "target&value"
    
    def test_chart_data_json_format_in_template_rendering(self) -> None:
        """Test that chart_data_json is correctly embedded in template output."""
        import json
        import re
        
        template = self._load_template()
        assigns = self._create_minimal_template_assigns()
        
        # Create multiple evidence points
        points = [
            EvidencePoint(
                step=i,
                evidences={"object1": 0.5 + i * 0.1, "object2": 0.3 + i * 0.05},
                target_object="object1",
                episode=i // 10,
            )
            for i in range(2)
        ]
        from collections import deque
        evidence_history = deque(points)
        
        # Update visualization state
        from contrib.liveview_experiment.src.visualization_state import VisualizationState
        from markupsafe import Markup
        viz_state = VisualizationState(evidence_history=evidence_history)
        assigns["chart_data_json"] = Markup(viz_state.get_chart_data_json())
        assigns["chart_point_count"] = len(points)
        
        # Render template
        result = template.render(**assigns)
        
        # Verify script tag is used
        assert 'script type="application/json"' in result or 'id="chart-data"' in result
        
        # Extract JSON from script tag
        pattern = r'<script[^>]*id="chart-data"[^>]*>(.*?)</script>'
        matches = re.findall(pattern, result, re.DOTALL)
        
        # Should have JSON content
        assert len(matches) > 0, "Expected chart-data script tag with JSON content"
        
        # Parse JSON - need to unescape HTML entities first
        json_content = matches[0].strip()
        # Unescape HTML entities (template engine may escape even with Markup)
        import html
        json_content = html.unescape(json_content)
        # If still has entities, unescape again (might be double-escaped)
        if '&quot;' in json_content or '&lt;' in json_content or '&gt;' in json_content:
            json_content = html.unescape(json_content)
        parsed = json.loads(json_content)
        
        # Verify structure
        assert "evidence_history" in parsed
        assert len(parsed["evidence_history"]) == len(points)
        
        # Verify each point structure
        for i, point in enumerate(parsed["evidence_history"]):
            assert "step" in point
            assert "evidences" in point
            assert isinstance(point["evidences"], dict)
            assert isinstance(point["step"], int)
            assert point["step"] == i

