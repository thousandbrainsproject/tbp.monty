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
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit

    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False
    print("Warning: radon is not installed. Complexity analysis will be limited.", file=sys.stderr)
    print("Install it with: pip install radon", file=sys.stderr)


@dataclass(frozen=True)
class ProtocolContext:
    """Context information about protocols in a file."""

    protocol_classes: Set[str]
    protocol_implementing_classes: Set[str]
    protocol_signatures: Dict[str, Set[str]]
    is_protocol_file: bool


@dataclass(frozen=True)
class AnalysisContext:
    """Context for analysis tools (radon, etc.)."""

    radon_results: List[Any]
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


def count_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Tuple[int, bool, bool]:
    """Count function parameters."""
    args = node.args
    regular_count = 0
    has_varargs = args.vararg is not None
    has_kwargs = args.kwarg is not None
    
    for arg in args.args:
        if arg.arg not in ('self', 'cls'):
            regular_count += 1
    
    return regular_count, has_varargs, has_kwargs


def is_protocol_class(node: ast.ClassDef, tree: ast.AST) -> bool:
    """Check if a class is a Protocol class."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "Protocol":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Protocol":
            if isinstance(base.value, ast.Name) and base.value.id in ("typing", "typing_extensions"):
                return True
    return False


def get_protocol_base_names(node: ast.ClassDef) -> List[str]:
    """Get names of Protocol classes that this class inherits from."""
    protocol_names = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            protocol_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            protocol_names.append(base.attr)
    return protocol_names


def find_parent_class(node: ast.AST, tree: ast.AST) -> Optional[ast.ClassDef]:
    """Find the parent class of a node."""
    for class_node in ast.walk(tree):
        if isinstance(class_node, ast.ClassDef):
            for child in ast.walk(class_node):
                if child is node:
                    return class_node
    return None


def collect_protocol_classes_from_file(
    tree: ast.AST, protocol_signatures: Dict[str, Set[str]]
) -> Set[str]:
    """Collect Protocol classes defined in this file."""
    protocol_classes: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and is_protocol_class(node, tree):
            protocol_classes.add(node.name)
            method_names = {
                child.name
                for child in ast.walk(node)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            protocol_signatures[node.name] = method_names
    return protocol_classes


def find_protocol_implementing_classes(
    tree: ast.AST, protocol_classes: Set[str], protocol_signatures: Dict[str, Set[str]]
) -> Set[str]:
    """Find classes that implement protocols."""
    implementing_classes: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            base_names = get_protocol_base_names(node)
            if any(base_name in protocol_classes or base_name in protocol_signatures for base_name in base_names):
                implementing_classes.add(node.name)
    return implementing_classes


def check_if_protocol_method(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    parent_class: Optional[ast.ClassDef],
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


def get_cyclomatic_complexity(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    func_name: str,
    line_start: int,
    radon_results: List[Any],
) -> int:
    """Get cyclomatic complexity for a function."""
    if RADON_AVAILABLE:
        for radon_func in radon_results:
            if radon_func.name == func_name and radon_func.lineno == line_start:
                return radon_func.complexity
    
    # Fallback: estimate from AST
    control_flow_types = (
        ast.If, ast.For, ast.While, ast.Try, ast.With,
        ast.AsyncFor, ast.AsyncWith,
    )
    control_flow_count = sum(1 for n in ast.walk(func_node) if isinstance(n, control_flow_types))
    return 1 + control_flow_count


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


def get_radon_metrics(source_code: str, file_path: Path) -> Tuple[List[Any], float]:
    """Get radon metrics if available."""
    radon_results: List[Any] = []
    mi_score = 0.0
    
    if not RADON_AVAILABLE:
        return radon_results, mi_score
    
    try:
        radon_results = cc_visit(source_code)
    except Exception as e:
        print(f"Warning: Radon error for {file_path}: {e}", file=sys.stderr)
    
    try:
        mi_result = mi_visit(source_code, multi=True)
        mi_score = mi_result[1] if isinstance(mi_result, tuple) else mi_result
    except Exception:
        pass
    
    return radon_results, mi_score


def analyze_file(file_path: Path, protocol_signatures: Optional[Dict[str, Set[str]]] = None) -> List[FunctionMetrics]:
    """Analyze a Python file and return function metrics."""
    try:
        source_code = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return []

    try:
        tree = ast.parse(source_code, filename=str(file_path))
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}", file=sys.stderr)
        return []

    if protocol_signatures is None:
        protocol_signatures = {}
    
    is_protocol_file = "protocol" in file_path.name.lower() or "contracts" in file_path.name.lower()
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
    
    metrics: List[FunctionMetrics] = []
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


def print_priority_table(top_metrics: List[FunctionMetrics], limit: int = 30) -> None:
    """Print top priority functions in a tabular format."""
    if not top_metrics:
        return

    project_root = Path(__file__).parent.parent.parent
    metrics_with_paths = []
    for m in top_metrics[:limit]:
        try:
            rel_path = Path(m.file_path).relative_to(project_root)
        except ValueError:
            rel_path = Path(m.file_path)
        metrics_with_paths.append((rel_path, m))
    
    max_file_len = min(max(len(str(p)) for p, _ in metrics_with_paths), 50)
    max_func_len = min(max(len(m.function_name) for _, m in metrics_with_paths), 30)
    
    col_widths = {
        'priority': 12,
        'file': max_file_len + 2,
        'function': max_func_len + 2,
        'lines': 12,
        'nest': 6,
        'complex': 8,
        'length': 8,
        'params': 8,
    }
    
    header = (
        f"{'Priority':<{col_widths['priority']}} "
        f"{'File':<{col_widths['file']}} "
        f"{'Function':<{col_widths['function']}} "
        f"{'Lines':<{col_widths['lines']}} "
        f"{'Nest':<{col_widths['nest']}} "
        f"{'Complex':<{col_widths['complex']}} "
        f"{'Length':<{col_widths['length']}} "
        f"{'Params':<{col_widths['params']}}"
    )
    separator = "=" * len(header)
    print(f"\n{separator}")
    print(header)
    print(separator)

    for rel_path, metric in metrics_with_paths:
        file_str = str(rel_path)[:col_widths['file'] - 2]
        func_str = metric.function_name[:col_widths['function'] - 2]
        param_str = str(metric.parameter_count)
        if metric.has_varargs:
            param_str += "+*args"
        if metric.has_kwargs:
            param_str += "+**kwargs"
        
        print(
            f"{format_priority(metric.priority_score):<{col_widths['priority']}} "
            f"{file_str:<{col_widths['file']}} "
            f"{func_str:<{col_widths['function']}} "
            f"{metric.line_start}-{metric.line_end:<{col_widths['lines']}} "
            f"{metric.max_nesting_level:<{col_widths['nest']}} "
            f"{metric.cyclomatic_complexity:<{col_widths['complex']}} "
            f"{metric.function_length:<{col_widths['length']}} "
            f"{param_str:<{col_widths['params']}}"
        )

    print(separator)
    print(f"\nShowing top {min(limit, len(top_metrics))} functions by priority score")


def _count_violations(metrics: List[FunctionMetrics], violation_type: str) -> int:
    """Count violations of a specific type."""
    if violation_type == "nesting":
        return sum(1 for m in metrics if m.max_nesting_level > 2)
    if violation_type == "complexity":
        return sum(1 for m in metrics if m.cyclomatic_complexity > 10)
    if violation_type == "length":
        return sum(1 for m in metrics if m.function_length > 50)
    if violation_type == "parameters":
        return sum(1 for m in metrics if m.parameter_violation > 0)
    return 0


def print_summary(
    all_metrics: List[FunctionMetrics],
    regular_metrics: List[FunctionMetrics],
    protocol_metrics: List[FunctionMetrics],
) -> None:
    """Print summary statistics."""
    print(f"\nTotal functions analyzed: {len(all_metrics)}")
    print(f"  - Regular functions: {len(regular_metrics)}")
    print(f"  - Protocol/interface methods: {len(protocol_metrics)}")
    print(f"Functions with nesting > 2: {_count_violations(regular_metrics, 'nesting')}")
    print(f"Functions with complexity > 10: {_count_violations(regular_metrics, 'complexity')}")
    print(f"Functions with length > 50: {_count_violations(regular_metrics, 'length')}")
    print(f"Functions with too many parameters: {_count_violations(regular_metrics, 'parameters')}")


def collect_protocol_signatures(root_dir: Path) -> Dict[str, Set[str]]:
    """Collect all Protocol class names and their method signatures."""
    protocol_signatures: Dict[str, Set[str]] = {}
    
    for py_file in find_python_files(root_dir):
        try:
            source_code = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source_code, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and is_protocol_class(node, tree):
                    method_names = {
                        child.name
                        for child in ast.walk(node)
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    }
                    protocol_signatures[node.name] = method_names
        except Exception:
            continue
    
    return protocol_signatures


def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path(__file__).parent.parent / "src"

    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing Python files in: {root_dir}")
    print("=" * 80)

    print("Collecting protocol signatures...")
    protocol_signatures = collect_protocol_signatures(root_dir)

    all_metrics: List[FunctionMetrics] = []
    for py_file in find_python_files(root_dir):
        metrics = analyze_file(py_file, protocol_signatures)
        all_metrics.extend(metrics)

    all_metrics.sort(key=lambda m: m.priority_score, reverse=True)

    protocol_metrics = [m for m in all_metrics if m.is_protocol_method]
    regular_metrics = [m for m in all_metrics if not m.is_protocol_method]

    print_summary(all_metrics, regular_metrics, protocol_metrics)

    print("\n" + "=" * 80)
    print("TOP REFACTORING PRIORITIES")
    print("=" * 80)
    top_priority_metrics = [m for m in regular_metrics if m.priority_score > 0]
    print_priority_table(top_priority_metrics, limit=30)

    report_path = Path(__file__).parent.parent / "complexity_report.json"
    report_data = {
        "summary": {
            "total_functions": len(all_metrics),
            "regular_functions": len(regular_metrics),
            "protocol_methods": len(protocol_metrics),
            "functions_with_nesting_violations": _count_violations(regular_metrics, "nesting"),
            "functions_with_high_complexity": _count_violations(regular_metrics, "complexity"),
            "functions_with_high_length": _count_violations(regular_metrics, "length"),
            "functions_with_too_many_parameters": _count_violations(regular_metrics, "parameters"),
        },
        "functions": [
            {
                "file": m.file_path,
                "function": m.function_name,
                "line_start": m.line_start,
                "line_end": m.line_end,
                "cyclomatic_complexity": m.cyclomatic_complexity,
                "max_nesting_level": m.max_nesting_level,
                "function_length": m.function_length,
                "parameter_count": m.parameter_count,
                "priority_score": m.priority_score,
            }
            for m in all_metrics
            if m.priority_score > 0 or (m.is_protocol_method and m.parameter_violation > 0)
        ],
    }
    
    with report_path.open("w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()

