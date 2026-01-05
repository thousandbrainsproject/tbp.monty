#!/usr/bin/env python3
"""Static analysis tool to measure code complexity and prioritize refactoring.

Adapted from my_mvg_departures for the liveview_experiment module.
Measures:
- Cyclomatic complexity (via radon)
- Maximum nesting level per function
- Function length (lines of code)
- Parameter count (max 4 regular params + *args + **kwargs)
- Overall priority score for refactoring
"""

from __future__ import annotations

import ast
import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit

    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False
    print(
        "Warning: radon is not installed. Complexity analysis will be limited.",
        file=sys.stderr,
    )
    print("Install it with: pip install radon", file=sys.stderr)


@dataclass(frozen=True)
class ProtocolContext:
    """Context information about protocols in a file."""

    protocol_classes: set[str]
    protocol_implementing_classes: set[str]
    protocol_signatures: dict[str, set[str]]
    is_protocol_file: bool


@dataclass(frozen=True)
class AnalysisContext:
    """Context for analysis tools (radon, etc.)."""

    radon_results: list[Any]
    mi_score: float


@dataclass(frozen=True)
class FileAnalysisContext:
    """Complete context for analyzing a file."""

    file_path: Path
    tree: ast.AST
    protocol_context: ProtocolContext
    analysis_context: AnalysisContext


@dataclass
class FunctionMetrics:
    """Metrics for a single function."""

    file_path: str
    function_name: str
    line_start: int
    line_end: int
    cyclomatic_complexity: int
    max_nesting_level: int
    function_length: int
    parameter_count: int
    has_varargs: bool
    has_kwargs: bool
    maintainability_index: float
    is_protocol_method: bool = False

    @property
    def parameter_violation(self) -> int:
        """Calculate parameter count violation (0 if OK, positive if too many)."""
        # Max allowed: 4 regular params + *args + **kwargs
        max_allowed = 4
        if self.has_varargs:
            max_allowed += 1
        if self.has_kwargs:
            max_allowed += 1
        return max(0, self.parameter_count - max_allowed)

    def _calculate_nesting_penalty(self) -> float:
        """Calculate penalty for nesting violations."""
        return max(0, (self.max_nesting_level - 2) * 10)

    def _calculate_complexity_penalty(self) -> float:
        """Calculate penalty for complexity violations."""
        return max(0, (self.cyclomatic_complexity - 10) * 2)

    def _calculate_length_penalty(self) -> float:
        """Calculate penalty for function length violations."""
        return max(0, (self.function_length - 50) * 0.5)

    def _calculate_parameter_penalty(self) -> float:
        """Calculate penalty for parameter count violations."""
        return self.parameter_violation * 3

    def _calculate_mi_penalty(self) -> float:
        """Calculate penalty based on Maintainability Index."""
        if self.maintainability_index <= 0:
            return 2.5
        if self.maintainability_index < 10:
            return 10 - (self.maintainability_index * 0.5)
        if self.maintainability_index < 20:
            return 5 - ((self.maintainability_index - 10) * 0.25)
        return 0.0

    @property
    def priority_score(self) -> float:
        """Calculate priority score for refactoring (higher = more urgent)."""
        return (
            self._calculate_nesting_penalty()
            + self._calculate_complexity_penalty()
            + self._calculate_length_penalty()
            + self._calculate_parameter_penalty()
            + self._calculate_mi_penalty()
        )


class NestingLevelVisitor(ast.NodeVisitor):
    """AST visitor to measure maximum nesting level in a function."""

    def __init__(self) -> None:
        """Initialize the visitor."""
        self.max_nesting = 0
        self.current_nesting = 0

    def visit(self, node: ast.AST) -> None:
        """Visit a node and track nesting."""
        if isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.Try,
                ast.With,
                ast.AsyncFor,
                ast.AsyncWith,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            self.current_nesting += 1
            self.max_nesting = max(self.max_nesting, self.current_nesting)
            self.generic_visit(node)
            self.current_nesting -= 1
        else:
            self.generic_visit(node)


def count_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, bool, bool]:
    """Count function parameters."""
    args = node.args
    regular_count = 0
    has_varargs = args.vararg is not None
    has_kwargs = args.kwarg is not None

    for arg in args.args:
        if arg.arg not in ("self", "cls"):
            regular_count += 1

    return regular_count, has_varargs, has_kwargs


def is_protocol_class(node: ast.ClassDef, _tree: ast.AST) -> bool:
    """Check if a class is a Protocol class."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "Protocol":
            return True
        if (
            isinstance(base, ast.Attribute)
            and base.attr == "Protocol"
            and isinstance(base.value, ast.Name)
            and base.value.id in ("typing", "typing_extensions")
        ):
            return True
    return False


def _extract_base_name(base: ast.expr) -> str | None:
    """Extract name from a base class expression.

    Args:
        base: Base class AST node

    Returns:
        Base name or None if not extractable
    """
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def get_protocol_base_names(node: ast.ClassDef) -> list[str]:
    """Get names of Protocol classes that this class inherits from."""
    protocol_names = []
    for base in node.bases:
        name = _extract_base_name(base)
        if name:
            protocol_names.append(name)
    return protocol_names


def _is_node_in_class(node: ast.AST, class_node: ast.ClassDef) -> bool:
    """Check if a node is contained within a class.

    Args:
        node: Node to find
        class_node: Class to search in

    Returns:
        True if node is in class
    """
    return any(child is node for child in ast.walk(class_node))


def find_parent_class(node: ast.AST, tree: ast.AST) -> ast.ClassDef | None:
    """Find the parent class of a node."""
    for class_node in ast.walk(tree):
        if isinstance(class_node, ast.ClassDef) and _is_node_in_class(node, class_node):
            return class_node
    return None


def _extract_method_names(node: ast.ClassDef) -> set[str]:
    """Extract method names from a class node.

    Args:
        node: Class definition node

    Returns:
        Set of method names
    """
    return {
        child.name
        for child in ast.walk(node)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def collect_protocol_classes_from_file(
    tree: ast.AST, protocol_signatures: dict[str, set[str]]
) -> set[str]:
    """Collect Protocol classes defined in this file."""
    protocol_classes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and is_protocol_class(node, tree):
            protocol_classes.add(node.name)
            protocol_signatures[node.name] = _extract_method_names(node)
    return protocol_classes


def _implements_protocol(
    base_names: list[str],
    protocol_classes: set[str],
    protocol_signatures: dict[str, set[str]],
) -> bool:
    """Check if base names indicate protocol implementation.

    Args:
        base_names: List of base class names
        protocol_classes: Set of protocol class names
        protocol_signatures: Dict of protocol signatures

    Returns:
        True if any base name is a protocol
    """
    return any(
        base_name in protocol_classes or base_name in protocol_signatures
        for base_name in base_names
    )


def find_protocol_implementing_classes(
    tree: ast.AST, protocol_classes: set[str], protocol_signatures: dict[str, set[str]]
) -> set[str]:
    """Find classes that implement protocols."""
    implementing_classes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            base_names = get_protocol_base_names(node)
            if _implements_protocol(base_names, protocol_classes, protocol_signatures):
                implementing_classes.add(node.name)
    return implementing_classes


def check_if_protocol_method(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    parent_class: ast.ClassDef | None,
    protocol_context: ProtocolContext,
) -> bool:
    """Check if a function is a protocol method."""
    if not parent_class:
        return protocol_context.is_protocol_file

    if parent_class.name in protocol_context.protocol_classes:
        return True

    if parent_class.name in protocol_context.protocol_implementing_classes:
        func_name = func_node.name
        return any(
            func_name in method_names
            for method_names in protocol_context.protocol_signatures.values()
        )
    return False


def _find_radon_complexity(
    func_name: str, line_start: int, radon_results: list[Any]
) -> int | None:
    """Find cyclomatic complexity from radon results.

    Args:
        func_name: Function name
        line_start: Starting line number
        radon_results: Radon analysis results

    Returns:
        Complexity value or None if not found
    """
    for radon_func in radon_results:
        if radon_func.name == func_name and radon_func.lineno == line_start:
            return radon_func.complexity
    return None


def _estimate_complexity_from_ast(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    """Estimate cyclomatic complexity from AST.

    Args:
        func_node: Function AST node

    Returns:
        Estimated complexity
    """
    control_flow_types = (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncFor,
        ast.AsyncWith,
    )
    control_flow_count = sum(
        1 for n in ast.walk(func_node) if isinstance(n, control_flow_types)
    )
    return 1 + control_flow_count


def get_cyclomatic_complexity(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    func_name: str,
    line_start: int,
    radon_results: list[Any],
) -> int:
    """Get cyclomatic complexity for a function."""
    if RADON_AVAILABLE:
        complexity = _find_radon_complexity(func_name, line_start, radon_results)
        if complexity is not None:
            return complexity

    return _estimate_complexity_from_ast(func_node)


def analyze_function_node(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    context: FileAnalysisContext,
) -> FunctionMetrics:
    """Analyze a single function node and return metrics."""
    func_name = func_node.name
    line_start = func_node.lineno
    line_end = func_node.end_lineno if hasattr(func_node, "end_lineno") else line_start
    function_length = line_end - line_start + 1

    parent_class = find_parent_class(func_node, context.tree)
    is_protocol_method = check_if_protocol_method(
        func_node, parent_class, context.protocol_context
    )

    visitor = NestingLevelVisitor()
    visitor.visit(func_node)
    max_nesting = visitor.max_nesting

    param_count, has_varargs, has_kwargs = count_parameters(func_node)
    cyclomatic_complexity = get_cyclomatic_complexity(
        func_node, func_name, line_start, context.analysis_context.radon_results
    )

    return FunctionMetrics(
        file_path=str(context.file_path),
        function_name=func_name,
        line_start=line_start,
        line_end=line_end,
        cyclomatic_complexity=cyclomatic_complexity,
        max_nesting_level=max_nesting,
        function_length=function_length,
        parameter_count=param_count,
        has_varargs=has_varargs,
        has_kwargs=has_kwargs,
        maintainability_index=float(context.analysis_context.mi_score),
        is_protocol_method=is_protocol_method,
    )


def get_radon_metrics(source_code: str, file_path: Path) -> tuple[list[Any], float]:
    """Get radon metrics if available."""
    radon_results: list[Any] = []
    mi_score = 0.0

    if not RADON_AVAILABLE:
        return radon_results, mi_score

    try:
        radon_results = cc_visit(source_code)
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Radon error for {file_path}: {e}", file=sys.stderr)

    with contextlib.suppress(Exception):
        mi_result = mi_visit(source_code, multi=True)
        mi_score = mi_result[1] if isinstance(mi_result, tuple) else mi_result

    return radon_results, mi_score


def analyze_file(
    file_path: Path, protocol_signatures: dict[str, set[str]] | None = None
) -> list[FunctionMetrics]:
    """Analyze a Python file and return function metrics."""
    try:
        source_code = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return []

    try:
        tree = ast.parse(source_code, filename=str(file_path))
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}", file=sys.stderr)
        return []

    if protocol_signatures is None:
        protocol_signatures = {}

    is_protocol_file = (
        "protocol" in file_path.name.lower() or "contracts" in file_path.name.lower()
    )
    protocol_classes = collect_protocol_classes_from_file(tree, protocol_signatures)
    protocol_implementing_classes = find_protocol_implementing_classes(
        tree, protocol_classes, protocol_signatures
    )

    radon_results, mi_score = get_radon_metrics(source_code, file_path)

    protocol_context = ProtocolContext(
        protocol_classes=protocol_classes,
        protocol_implementing_classes=protocol_implementing_classes,
        protocol_signatures=protocol_signatures,
        is_protocol_file=is_protocol_file,
    )
    analysis_context = AnalysisContext(
        radon_results=radon_results,
        mi_score=mi_score,
    )
    file_context = FileAnalysisContext(
        file_path=file_path,
        tree=tree,
        protocol_context=protocol_context,
        analysis_context=analysis_context,
    )

    metrics: list[FunctionMetrics] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            metric = analyze_function_node(node, file_context)
            metrics.append(metric)

    return metrics


def find_python_files(root_dir: Path) -> Iterator[Path]:
    """Find all Python files in the source directory."""
    for path in root_dir.rglob("*.py"):
        if "__pycache__" not in str(path) and "test_" not in path.name:
            yield path


def format_priority(score: float) -> str:
    """Format priority score as a string."""
    if score >= 20:
        return "🔴 CRITICAL"
    if score >= 10:
        return "🟠 HIGH"
    if score >= 5:
        return "🟡 MEDIUM"
    if score > 0:
        return "🟢 LOW"
    return "✅ OK"


def _prepare_metrics_paths(
    top_metrics: list[FunctionMetrics], limit: int
) -> list[tuple[Path, FunctionMetrics]]:
    """Prepare metrics with relative paths.

    Args:
        top_metrics: List of function metrics
        limit: Maximum number of metrics to process

    Returns:
        List of (relative_path, metric) tuples
    """
    project_root = Path(__file__).parent.parent.parent
    metrics_with_paths = []
    for m in top_metrics[:limit]:
        try:
            rel_path = Path(m.file_path).relative_to(project_root)
        except ValueError:
            rel_path = Path(m.file_path)
        metrics_with_paths.append((rel_path, m))
    return metrics_with_paths


def print_priority_table(top_metrics: list[FunctionMetrics], limit: int = 30) -> None:
    """Print top priority functions in a tabular format."""
    if not top_metrics:
        return

    from .table_formatter import TableFormatter  # noqa: PLC0415

    metrics_with_paths = _prepare_metrics_paths(top_metrics, limit)
    col_widths = TableFormatter.calculate_column_widths(metrics_with_paths)
    header = TableFormatter.build_header(col_widths)
    separator = "=" * len(header)

    print(f"\n{separator}")
    print(header)
    print(separator)

    for rel_path, metric in metrics_with_paths:
        row = TableFormatter.format_row(rel_path, metric, col_widths, format_priority)
        print(row)

    print(separator)
    print(f"\nShowing top {min(limit, len(top_metrics))} functions by priority score")


def _count_violations(metrics: list[FunctionMetrics], violation_type: str) -> int:
    """Count violations of a specific type."""
    violation_checks = {
        "nesting": lambda m: m.max_nesting_level > 2,
        "complexity": lambda m: m.cyclomatic_complexity > 10,
        "length": lambda m: m.function_length > 50,
        "parameters": lambda m: m.parameter_violation > 0,
    }
    check = violation_checks.get(violation_type)
    return sum(1 for m in metrics if check(m)) if check else 0


def print_summary(
    all_metrics: list[FunctionMetrics],
    regular_metrics: list[FunctionMetrics],
    protocol_metrics: list[FunctionMetrics],
) -> None:
    """Print summary statistics."""
    print(f"\nTotal functions analyzed: {len(all_metrics)}")
    print(f"  - Regular functions: {len(regular_metrics)}")
    print(f"  - Protocol/interface methods: {len(protocol_metrics)}")
    print(
        f"Functions with nesting > 2: {_count_violations(regular_metrics, 'nesting')}"
    )
    complexity_count = _count_violations(regular_metrics, "complexity")
    print(f"Functions with complexity > 10: {complexity_count}")
    length_count = _count_violations(regular_metrics, "length")
    print(f"Functions with length > 50: {length_count}")
    param_count = _count_violations(regular_metrics, "parameters")
    print(f"Functions with too many parameters: {param_count}")


def _extract_protocol_from_file(
    py_file: Path, protocol_signatures: dict[str, set[str]]
) -> None:
    """Extract protocol signatures from a single file.

    Args:
        py_file: Python file path
        protocol_signatures: Dict to update with protocol signatures
    """
    try:
        source_code = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source_code, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and is_protocol_class(node, tree):
                protocol_signatures[node.name] = _extract_method_names(node)
    except (SyntaxError, UnicodeDecodeError):
        pass


def collect_protocol_signatures(root_dir: Path) -> dict[str, set[str]]:
    """Collect all Protocol class names and their method signatures."""
    protocol_signatures: dict[str, set[str]] = {}
    for py_file in find_python_files(root_dir):
        _extract_protocol_from_file(py_file, protocol_signatures)
    return protocol_signatures


def _get_root_directory() -> Path:
    """Get root directory for analysis.

    Returns:
        Root directory path
    """
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return Path(__file__).parent.parent / "src"


def _collect_all_metrics(
    root_dir: Path, protocol_signatures: dict[str, set[str]]
) -> list[FunctionMetrics]:
    """Collect metrics from all Python files.

    Args:
        root_dir: Root directory to analyze
        protocol_signatures: Protocol signature dictionary

    Returns:
        List of all function metrics
    """
    all_metrics: list[FunctionMetrics] = []
    for py_file in find_python_files(root_dir):
        metrics = analyze_file(py_file, protocol_signatures)
        all_metrics.extend(metrics)
    all_metrics.sort(key=lambda m: m.priority_score, reverse=True)
    return all_metrics


def _separate_metrics(
    all_metrics: list[FunctionMetrics],
) -> tuple[list[FunctionMetrics], list[FunctionMetrics]]:
    """Separate metrics into protocol and regular.

    Args:
        all_metrics: All function metrics

    Returns:
        Tuple of (protocol_metrics, regular_metrics)
    """
    protocol_metrics = [m for m in all_metrics if m.is_protocol_method]
    regular_metrics = [m for m in all_metrics if not m.is_protocol_method]
    return protocol_metrics, regular_metrics


def main() -> None:
    """Main entry point."""
    root_dir = _get_root_directory()

    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing Python files in: {root_dir}")
    print("=" * 80)

    print("Collecting protocol signatures...")
    protocol_signatures = collect_protocol_signatures(root_dir)

    all_metrics = _collect_all_metrics(root_dir, protocol_signatures)
    protocol_metrics, regular_metrics = _separate_metrics(all_metrics)

    print_summary(all_metrics, regular_metrics, protocol_metrics)

    print("\n" + "=" * 80)
    print("TOP REFACTORING PRIORITIES")
    print("=" * 80)
    top_priority_metrics = [m for m in regular_metrics if m.priority_score > 0]
    print_priority_table(top_priority_metrics, limit=30)

    from .report_generator import ReportGenerator  # noqa: PLC0415

    report_path = Path(__file__).parent.parent / "complexity_report.json"
    report_data = ReportGenerator.build_report_data(
        all_metrics, regular_metrics, protocol_metrics, _count_violations
    )
    ReportGenerator.save_report(report_path, report_data)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
