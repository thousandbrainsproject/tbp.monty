# Copyright 2024 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Generate proto/tbp/simulator/protocol/v1/protocol.proto.

Generate proto/tbp/simulator/protocol/v1/protocol.proto from
the Python Protocol at src/tbp/monty/simulators/simulator.py:Simulator.

Usage:
  python tools/generate_simulator_proto/cli.py
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---- Python 3.8 compatibility: prefer stdlib, fall back to typing_extensions
try:
    from typing import Annotated, get_args, get_origin
except ImportError:  # Python 3.8 may not have all of these in typing
    from typing_extensions import (  # type: ignore[import-untyped]
        Annotated,
        get_args,
        get_origin,
    )

# ---------------- Configuration (edit if you move things) ---------------------

# Python symbol of the Protocol to convert into a gRPC service
# This loader will import the symbol below; we also add repo_root/src to sys.path.
SIMULATOR_SYMBOL = "tbp.monty.simulators.simulator:Simulator"

# Protobuf package & output file
PROTO_PACKAGE = "tbp.simulator.protocol.v1"
PROTO_OUT = Path("proto/tbp/simulator/protocol/v1/protocol.proto")

# -----------------------------------------------------------------------------


# Simple metadata hook (kept for future expansion: explicit tags/json_names)
class Pb:
    """Simple metadata hook for protobuf field annotations."""

    def __init__(
        self,
        tag: Optional[int] = None,
        json_name: Optional[str] = None,
        comment: Optional[str] = None,
    ):
        """Initialize Pb metadata.

        Args:
            tag: Optional field tag number
            json_name: Optional JSON field name
            comment: Optional field comment
        """
        self.tag = tag
        self.json_name = json_name
        self.comment = comment


_SCALAR_MAP = {
    str: "string",
    int: "int64",   # conservative default
    float: "double",
    bool: "bool",
    bytes: "bytes",
}

_EMPTY_IMPORT = 'import "google/protobuf/empty.proto";'
_TIMESTAMP_IMPORT = 'import "google/protobuf/timestamp.proto";'


def _ensure_src_on_syspath() -> None:
    """Ensure repo_root/src is on sys.path so tbp.* imports resolve."""
    here = Path(__file__).resolve()
    # tools/generate_simulator_proto/cli.py -> repo root is parents[2]
    repo_root = here.parents[2]
    src = repo_root / "src"
    if src.is_dir():
        p = str(src)
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_symbol(qualified: str) -> Any:
    """Load a symbol specified as 'pkg.module:Name'.

    Args:
        qualified: Symbol specification in format 'pkg.module:Name'

    Returns:
        The loaded symbol

    Raises:
        ValueError: If qualified string doesn't contain ':'
    """
    if ":" not in qualified:
        raise ValueError(f"Expected 'module:Symbol', got '{qualified}'")
    mod_name, sym_name = qualified.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


def _snake_to_pascal(snake_str: str) -> str:
    """Convert snake_case string to PascalCase.

    Args:
        snake_str: String in snake_case format

    Returns:
        String in PascalCase format
    """
    components = snake_str.split("_")
    return "".join(word.capitalize() for word in components)


def _is_optional(t: Any) -> Tuple[bool, Any]:
    origin = get_origin(t)
    # Handle typing.Union (e.g., Union[T, None])
    if origin is Union:
        args = tuple(get_args(t))
        non_none = tuple(a for a in args if a is not type(None))
        if (len(non_none) == 1 and len(args) == 2 and
            any(a is type(None) for a in args)):
            return True, non_none[0]

    # Handle Python 3.10+ union syntax (e.g., T | None)
    try:
        import types
        if hasattr(types, "UnionType") and isinstance(t, types.UnionType):
            args = tuple(t.__args__)
            non_none = tuple(a for a in args if a is not type(None))
            if (len(non_none) == 1 and len(args) == 2 and
                any(a is type(None) for a in args)):
                return True, non_none[0]
    except (ImportError, AttributeError):
        # Python < 3.10 doesn't have types.UnionType
        pass

    return False, t


def _iter_method_signatures(cls: Any) -> List[Tuple[str, inspect.Signature]]:
    """Collect public methods (in declared order) and their signatures.

    Args:
        cls: Class to inspect

    Returns:
        List of (method_name, signature) tuples
    """
    methods: List[Tuple[str, inspect.Signature]] = []
    for name, func in cls.__dict__.items():
        if name.startswith("_"):
            continue
        if not inspect.isfunction(func) and not inspect.ismethoddescriptor(func):
            continue
        sig = inspect.signature(func)
        methods.append((name, sig))
    return methods


@dataclasses.dataclass
class FieldSpec:
    name: str
    py_type: Any
    optional: bool = False
    repeated: bool = False
    is_map: bool = False
    tag: Optional[int] = None
    json_name: Optional[str] = None
    comment: Optional[str] = None


class ProtoBuilder:
    def __init__(self, package: str, existing_proto_path: Optional[Path] = None):
        self.package = package
        self.enums: Dict[str, str] = {}
        self.messages: Dict[str, str] = {}
        self.services: Dict[str, str] = {}
        self._seen_types: Dict[Any, str] = {}  # cache: py type -> proto type name
        self._need_empty = False
        self._need_timestamp = False
        self._declared_placeholders: set[str] = set()
        # Field number persistence
        # message_name -> {field_name: field_number}
        self._field_numbers: Dict[str, Dict[str, int]] = {}

        if existing_proto_path and existing_proto_path.exists():
            self._parse_existing_proto(existing_proto_path)

    def _parse_existing_proto(self, proto_path: Path) -> None:
        """Parse existing proto file to extract field numbers for persistence."""
        try:
            import subprocess
            import tempfile

            from google.protobuf import descriptor_pb2
        except ImportError:
            print(
                "Warning: google.protobuf not available. "
                "Install with: pip install protobuf"
            )
            self._field_numbers = {}
            return

        try:
            # Use protoc to compile the proto file to a descriptor
            with tempfile.NamedTemporaryFile(suffix=".desc", delete=False) as desc_file:
                desc_path = desc_file.name

            try:
                # Run protoc to generate a binary descriptor
                result = subprocess.run([
                    "protoc",
                    f"--descriptor_set_out={desc_path}",
                    f"--proto_path={proto_path.parent}",
                    str(proto_path)
                ], capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    # If protoc fails, fall back to simple text parsing
                    print(f"Warning: protoc failed, using fallback parsing: {result.stderr}")
                    self._parse_existing_proto_fallback(proto_path)
                    return

                # Read the binary descriptor
                with open(desc_path, "rb") as f:
                    file_descriptor_set = descriptor_pb2.FileDescriptorSet()
                    file_descriptor_set.ParseFromString(f.read())

                # Extract field numbers from the descriptor
                for file_descriptor in file_descriptor_set.file:
                    for message_type in file_descriptor.message_type:
                        message_name = message_type.name
                        if message_name not in self._field_numbers:
                            self._field_numbers[message_name] = {}

                        for field in message_type.field:
                            field_name = field.name
                            field_number = field.number
                            self._field_numbers[message_name][field_name] = field_number

            finally:
                # Clean up temp file
                try:
                    Path(desc_path).unlink()
                except OSError:
                    pass

        except (OSError, subprocess.TimeoutExpired, Exception) as e:
            print(f"Warning: Could not parse existing proto file {proto_path}: {e}")
            # Fall back to simple text parsing
            self._parse_existing_proto_fallback(proto_path)

    def _parse_existing_proto_fallback(self, proto_path: Path) -> None:
        """Fallback text-based parsing when protoc is not available."""
        try:
            content = proto_path.read_text(encoding="utf-8")
            current_message = None
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Check for message definition
                if line.startswith("message ") and line.endswith(" {"):
                    # Extract message name, handling invalid names
                    message_part = line[8:-2].strip()  # Remove "message " and " {"
                    # Only use valid protobuf identifiers
                    if message_part.replace("_", "").replace(".", "").isalnum():
                        current_message = message_part
                        if current_message not in self._field_numbers:
                            self._field_numbers[current_message] = {}
                    else:
                        current_message = None
                    continue
                
                # Check for end of message
                if line == "}":
                    current_message = None
                    continue
                
                # Parse field definitions
                if current_message and "=" in line and line.endswith(";"):
                    # Simple field parsing: type name = number;
                    parts = line[:-1].split("=")  # Remove semicolon
                    if len(parts) == 2:
                        field_part = parts[0].strip()
                        number_part = parts[1].strip()
                        
                        # Extract field name (last word before =)
                        field_words = field_part.split()
                        if field_words:
                            field_name = field_words[-1]
                            try:
                                field_number = int(number_part)
                                self._field_numbers[current_message][field_name] = field_number
                            except ValueError:
                                pass  # Skip invalid field numbers

        except OSError as e:
            print(f"Warning: Fallback parsing failed for {proto_path}: {e}")
            self._field_numbers = {}

    # ---------- Top-level render ----------
    def render_file(self) -> str:
        lines: List[str] = []
        lines.append('syntax = "proto3";\n')
        if self.package:
            lines.append(f"package {self.package};\n")
        if self._need_empty:
            lines.append(_EMPTY_IMPORT + "\n")
        if self._need_timestamp:
            lines.append(_TIMESTAMP_IMPORT + "\n")
        lines.append("\n")

        # enums, messages, services
        for body in self.enums.values():
            lines.append(body.rstrip() + "\n\n")
        for body in self.messages.values():
            lines.append(body.rstrip() + "\n\n")
        for body in self.services.values():
            lines.append(body.rstrip() + "\n\n")
        return "".join(lines)

    # ---------- Types ----------
    def type_to_proto(self, t: Any) -> str:
        # cache
        if t in self._seen_types:
            return self._seen_types[t]

        # Handle string type annotations (forward references)
        if isinstance(t, str):
            if t == "None":
                # This should not happen - indicates a bug in type processing
                msg = (
                    "String 'None' encountered as type annotation. "
                    "This suggests an issue with type resolution."
                )
                raise TypeError(msg)
            # Clean up the string to make it a valid protobuf identifier
            clean_name = self._make_valid_proto_name(t)
            self._emit_placeholder_message(clean_name)
            self._seen_types[t] = clean_name
            return clean_name

        # Handle None type explicitly
        if t is None or t is type(None):
            # None type should not reach here - should be handled at field level
            msg = (
                    "None type should be handled at field level, "
                "not passed to type_to_proto"
            )
            raise TypeError(msg)

        # scalars
        if t in _SCALAR_MAP:
            return _SCALAR_MAP[t]

        # special-case datetime -> Timestamp
        try:
            import datetime
            if t is datetime.datetime:
                self._need_timestamp = True
                return "google.protobuf.Timestamp"
        except ImportError:
            pass

        origin = get_origin(t)

        # Annotated[T, ...] -> T
        if origin is Annotated:
            base, *_ = get_args(t)
            return self.type_to_proto(base)

        # list[T] / set[T] / tuple[T] -> element type (repeated handled at field-level)
        if origin in (list, List, set, tuple):
            elem = get_args(t)[0] if get_args(t) else Any
            return self.type_to_proto(elem)

        # dict[str, V] -> map<string, V> (map handled at field-level)
        if origin in (dict, Dict):
            _, v = get_args(t) if get_args(t) else (str, Any)
            return self.type_to_proto(v)

        # Enums
        if inspect.isclass(t) and issubclass(t, enum.Enum):
            name = t.__name__
            if name not in self.enums:
                self._emit_enum_from_py_enum(t)
            self._seen_types[t] = name
            return name

        # Handle NewType instances by extracting the underlying type
        if hasattr(t, "__supertype__"):
            # This is a NewType - use the underlying type but keep the name
            name = getattr(t, "__name__", str(t))
            underlying = t.__supertype__
            underlying_proto = self.type_to_proto(underlying)
            self._seen_types[t] = underlying_proto
            return underlying_proto

        # Message (placeholder for unknown app-specific type)
        if inspect.isclass(t):
            name = self._make_valid_proto_name(t.__name__)
            if name not in self.messages:
                self._emit_placeholder_message(name)
            self._seen_types[t] = name
            return name

        # Fallback with better error info
        msg = f"Unsupported type in Protocol → proto mapping: {t!r} (type: {type(t)})"
        raise TypeError(msg)

    def _emit_enum_from_py_enum(self, e: type[enum.Enum]) -> None:
        # Require integers for proto enums
        if not all(isinstance(m.value, int) for m in e):
            raise TypeError(f"Enum {e.__name__} must use integer values for protobuf.")
        zero_seen = any(int(m.value) == 0 for m in e)
        lines = [f"enum {e.__name__} {{\n"]
        if not zero_seen:
            lines.append(f"  {e.__name__.upper()}_UNSPECIFIED = 0;\n")
        for m in e:
            lines.append(f"  {m.name} = {int(m.value)};\n")
        lines.append("}\n")
        self.enums[e.__name__] = "".join(lines)

    def _make_valid_proto_name(self, name: str) -> str:
        """Convert a Python type name to a valid protobuf identifier."""
        # Remove invalid characters and replace with underscore
        import re
        # Replace common Python type syntax with valid identifiers
        clean_name = name
        clean_name = re.sub(r'\s*\|\s*None\s*', 'Optional', clean_name)
        clean_name = re.sub(r'tuple\[([^\]]+)\]', r'Tuple_\1', clean_name)
        clean_name = re.sub(r'list\[([^\]]+)\]', r'List_\1', clean_name)
        clean_name = re.sub(r'dict\[([^\]]+)\]', r'Dict_\1', clean_name)
        # Remove all non-alphanumeric characters except underscore
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        # Ensure it starts with a letter
        if clean_name and clean_name[0].isdigit():
            clean_name = f"Type_{clean_name}"
        # Remove multiple consecutive underscores
        clean_name = re.sub(r'_{2,}', '_', clean_name)
        # Remove trailing/leading underscores
        clean_name = clean_name.strip('_')
        return clean_name or "UnknownType"

    def _emit_placeholder_message(self, name: str) -> None:
        # Avoid redefining placeholders
        if name in self._declared_placeholders:
            return
        body = textwrap.dedent(f"""\
        // Placeholder for application type '{name}'. Flesh out fields later.
        message {name} {{
        }}
        """)
        self.messages[name] = body
        self._declared_placeholders.add(name)

    # ---------- Message emission ----------
    def emit_message(self, name: str, fields: List[FieldSpec]) -> str:
        lines = [f"message {name} {{\n"]

        # Get existing field numbers for this message (only for fields currently in proto)
        existing_fields = self._field_numbers.get(name, {})

        # Find the next field number by taking the highest existing number + 1
        # This ensures we never reuse old field numbers (protobuf best practice)
        if existing_fields:
            next_number = max(existing_fields.values()) + 1
        else:
            next_number = 1

        # First pass: assign field numbers to all fields
        field_assignments = []  # List of (field_number, field_spec, proto_type)

        for f in fields:
            ptype = self.type_to_proto(f.py_type)

            # Only reuse field number if the field currently exists in the proto file
            # This ensures manually removed fields get new numbers when re-added
            if f.name in existing_fields:
                field_number = existing_fields[f.name]
            else:
                field_number = next_number
                # Update our tracking for future runs
                if name not in self._field_numbers:
                    self._field_numbers[name] = {}
                self._field_numbers[name][f.name] = field_number
                # Increment for next new field
                next_number += 1

            field_assignments.append((field_number, f, ptype))

        # Second pass: sort by field number and generate output
        field_assignments.sort(key=lambda x: x[0])  # Sort by field number
        
        for field_number, f, ptype in field_assignments:
            if f.is_map:
                lines.append(f"  map<string, {ptype}> {f.name} = {field_number};\n")
            else:
                repeated_prefix = "repeated " if f.repeated else ""
                optional_prefix = (
                    "optional "
                    if f.optional and ptype != "google.protobuf.Timestamp"
                    else ""
                )
                prefix = repeated_prefix + optional_prefix
                lines.append(f"  {prefix}{ptype} {f.name} = {field_number};\n")

        lines.append("}\n")
        body = "".join(lines)
        self.messages[name] = body
        return name

    # ---------- Service emission ----------
    def emit_service(self, name: str, rpcs: List[Tuple[str, str, str]]) -> None:
        """Emit a service definition.

        Args:
            name: Service name
            rpcs: List of (method_name, request_type, response_type) tuples
        """
        lines = [f"service {name} {{\n"]
        for m, req, resp in rpcs:
            lines.append(f"  rpc {m} ({req}) returns ({resp});\n")
        lines.append("}\n")
        self.services[name] = "".join(lines)


def _fields_from_parameters(params: List[inspect.Parameter]) -> List[FieldSpec]:
    fields: List[FieldSpec] = []
    for p in params:
        if p.name in ("self", "cls"):
            continue
        allowed_kinds = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        if p.kind not in allowed_kinds:
            msg = f"Unsupported parameter kind for protobuf RPC: {p.name} ({p.kind})"
            raise TypeError(msg)
        # Type (fallback to bytes if missing to avoid breakage)
        t = (
            p.annotation
            if p.annotation is not inspect._empty
            else bytes
        )
        # Optional?
        is_opt, inner = _is_optional(t)
        origin = get_origin(inner)
        repeated = origin in (list, List, set, tuple)
        is_map = origin in (dict, Dict)
        py_t = inner
        if repeated:
            py_t = get_args(inner)[0] if get_args(inner) else Any
        if is_map:
            _, v = get_args(inner) if get_args(inner) else (str, Any)
            py_t = v
        fields.append(FieldSpec(
            name=p.name,
            py_type=py_t,
            optional=is_opt and not repeated and not is_map,
            repeated=repeated,
            is_map=is_map,
        ))
    return fields


def _fields_from_return(
    builder: ProtoBuilder, ret: Any, method_name: str
) -> Tuple[Optional[str], Optional[List[FieldSpec]]]:
    """Convert return annotation to response message fields.

    Returns (response_message_name, fields) or (None, None) to indicate
    google.protobuf.Empty.
    - Treats no annotation OR explicit `-> None` as Empty.
    - For Tuple[...] returns, each element becomes a field:
        * Optional[T] -> optional T
        * literal None -> optional bytes (placeholder to avoid crashes)

    Args:
        builder: Proto builder instance
        ret: Return type annotation
        method_name: Name of the method

    Returns:
        Tuple of (message_name, fields) or (None, None) for Empty
    """
    # No return or explicit None -> Empty
    if ret is inspect._empty or ret is None or ret is type(None):
        builder._need_empty = True
        return None, None

    # Handle string type annotations
    if isinstance(ret, str):
        if ret == "None":
            builder._need_empty = True
            return None, None
        # For other strings, treat as a single field
        fields = [FieldSpec(name="result", py_type=ret, optional=False)]
        return f"{method_name}Response", fields

    # Optional[...] -> unwrap; Optional[None] -> Empty
    is_opt, inner = _is_optional(ret)
    if is_opt and inner is type(None):
        builder._need_empty = True
        return None, None
    ret = inner

    origin = get_origin(ret)

    # Tuple[...] -> message with multiple fields
    if origin in (tuple, Tuple):
        args = list(get_args(ret))
        fields: List[FieldSpec] = []
        for i, t in enumerate(args, start=1):
            # If a tuple slot is literally None (rare but can appear from
            # unresolved typing), generate an optional placeholder 'bytes' field
            # so codegen succeeds.
            if t is None or t is type(None):
                fields.append(FieldSpec(
                    name=f"result_{i}",
                    py_type=bytes,
                    optional=True,
                ))
                continue

            # Handle string type annotations in tuple elements
            if isinstance(t, str):
                if t == "None":
                    fields.append(FieldSpec(
                        name=f"result_{i}",
                        py_type=bytes,
                        optional=True,
                    ))
                    continue
                # For other strings, use as-is
                fields.append(FieldSpec(
                    name=f"result_{i}",
                    py_type=t,
                    optional=False,
                ))
                continue

            is_opt_i, inner_i = _is_optional(t)

            # If Optional[None] slipped through, also use placeholder
            if is_opt_i and inner_i is type(None):
                fields.append(FieldSpec(
                    name=f"result_{i}",
                    py_type=bytes,
                    optional=True,
                ))
                continue

            # Handle string inner types
            if isinstance(inner_i, str) and inner_i == "None":
                fields.append(FieldSpec(
                    name=f"result_{i}",
                    py_type=bytes,
                    optional=True,
                ))
                continue

            fields.append(FieldSpec(
                name=f"result_{i}",
                py_type=inner_i,
                optional=is_opt_i,
            ))
        return f"{method_name}Response", fields

    # Plain type -> single-field message
    fields = [FieldSpec(name="result", py_type=ret, optional=False)]
    return f"{method_name}Response", fields


def generate_from_simulator_protocol() -> str:
    # Load Protocol
    simulator = _load_symbol(SIMULATOR_SYMBOL)
    if not (isinstance(simulator, type) and getattr(simulator, "_is_protocol", False)):
        raise TypeError(f"{SIMULATOR_SYMBOL} is not a typing.Protocol")

    builder = ProtoBuilder(PROTO_PACKAGE, PROTO_OUT)

    # Build service from methods
    rpcs: List[Tuple[str, str, str]] = []

    for method_name, sig in _iter_method_signatures(simulator):
        # Convert method name to PascalCase for protobuf conventions
        pascal_method_name = _snake_to_pascal(method_name)
        # Request
        req_fields = _fields_from_parameters(list(sig.parameters.values()))
        if req_fields:
            req_msg = f"{pascal_method_name}Request"
            builder.emit_message(req_msg, req_fields)
        else:
            # No-arg request -> Empty
            builder._need_empty = True
            req_msg = "google.protobuf.Empty"

        # Response (by return annotation)
        ret_ann = sig.return_annotation
        resp_msg_name, resp_fields = _fields_from_return(
            builder, ret_ann, pascal_method_name
        )
        if resp_msg_name is None:
            resp = "google.protobuf.Empty"
        else:
            builder.emit_message(resp_msg_name, resp_fields or [])
            resp = resp_msg_name

        rpcs.append((pascal_method_name, req_msg, resp))

    # Emit the service
    builder.emit_service("Simulator", rpcs)

    # Done
    return builder.render_file()


def main() -> None:
    _ensure_src_on_syspath()
    proto_text = generate_from_simulator_protocol()
    PROTO_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROTO_OUT.write_text(proto_text, encoding="utf-8")
    rel = os.path.relpath(str(PROTO_OUT))
    print(f"Wrote {rel}")


if __name__ == "__main__":
    main()
