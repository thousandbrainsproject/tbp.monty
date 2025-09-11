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

import ast
import dataclasses
import datetime
import enum
import importlib
import inspect
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from google.protobuf import descriptor_pb2
from typing_extensions import Annotated, get_args, get_origin

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

# String name to scalar type mapping for string annotations
_SCALAR_NAME_MAP = {
    "str": "string",
    "int": "int64",
    "float": "double",
    "bool": "bool",
    "bytes": "bytes",
}

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


def _type_name_to_field_name_standalone(type_name: str) -> str:
    """Convert a type name to a snake_case field name."""
    # Convert PascalCase to snake_case
    # e.g., "ProprioceptiveState" -> "proprioceptive_state"
    # e.g., "ObjectID" -> "object_id"
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", type_name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


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
    # Handle string type annotations (forward references) with union syntax
    if isinstance(t, str):
        # Check for " | None" or "| None" patterns in string annotations
        if " | None" in t or "|None" in t:
            # Extract the non-None type from the string
            clean_type = re.sub(r"\s*\|\s*None\s*", "", t).strip()
            return True, clean_type
        return False, t

    origin = get_origin(t)

    # Handle typing.Union (e.g., Union[T, None])
    if origin is Union:
        args = tuple(get_args(t))
        non_none = tuple(a for a in args if a is not type(None))
        if (len(non_none) == 1 and len(args) == 2 and
            any(a is type(None) for a in args)):
            return True, non_none[0]

    # Handle Python 3.10+ union syntax (e.g., T | None)
    # Check if this is a union with None
    if hasattr(t, "__class__") and "UnionType" in str(type(t)):
        args = get_args(t)
        if args:
            non_none = tuple(a for a in args if a is not type(None))
            if (
                len(non_none) == 1
                and len(args) == 2
                and any(a is type(None) for a in args)
            ):
                return True, non_none[0]

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

        # Add language-specific package options
        lines.append("\n")
        lines.append('option go_package = "tbp/simulator/protocol/v1;protocolv1";\n')
        lines.append('option java_package = "org.thousandbrains.simulator.v1";\n')
        lines.append('option csharp_namespace = "ThousandBrains.Simulator.V1";\n')

        if self._need_timestamp:
            lines.append(_TIMESTAMP_IMPORT + "\n")

        # Add documentation comments
        lines.append("\n")
        lines.append("// All linear units in meters, angles in radians.\n")
        lines.append("// Right-handed coordinate system.\n")
        lines.append("// Quaternions use w,x,y,z (scalar-first) ordering.\n")
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
            # Check if this is a built-in scalar type
            if t in _SCALAR_NAME_MAP:
                return _SCALAR_NAME_MAP[t]

            # Handle ID types as strings
            if t.endswith("ID"):
                return "uint64"

            # Strip union part before trying to resolve (e.g., "VectorXYZ | None" -> "VectorXYZ")
            base_type_name = re.sub(r"\s*\|\s*None\s*", "", t).strip()

            # Try to resolve the base type by loading it
            resolved_type = self._resolve_string_type(base_type_name)
            if resolved_type is not None:
                return self.type_to_proto(resolved_type)

            # Handle special cases for generic type expressions
            if re.match(r"tuple\[", base_type_name, re.IGNORECASE):
                # Parse tuple expression like "tuple[A, B]" and create simplified message
                clean_name = self._make_valid_proto_name(base_type_name)
                self._create_tuple_message(base_type_name, clean_name)
                self._seen_types[t] = clean_name
                return clean_name

            # Clean up the string to make it a valid protobuf identifier
            clean_name = self._make_valid_proto_name(base_type_name)
            self._introspect_and_emit_message(resolved_type, clean_name)
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

        # Handle ID types - convert int-based IDs to uint64
        if hasattr(t, "__name__") and t.__name__.endswith("ID"):
            return "uint64"

        # special-case datetime -> Timestamp
        if t is datetime.datetime:
            self._need_timestamp = True
            return "google.protobuf.Timestamp"

        # Handle numpy arrays as bytes (standard protobuf type)
        if hasattr(t, "__module__") and t.__module__ and "numpy" in t.__module__:
            return "bytes"

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
            # This is a NewType - check if it's a simple scalar wrapper
            underlying = t.__supertype__
            if underlying in _SCALAR_MAP:
                # Simple scalar NewType (like ObjectID = NewType("ObjectID", int))
                proto_type = _SCALAR_MAP[underlying]
                self._seen_types[t] = proto_type
                return proto_type
            else:
                # Complex NewType - create a message for the NewType itself
                # (e.g., VectorXYZ = NewType("VectorXYZ", Tuple[float, float, float]))
                name = self._make_valid_proto_name(getattr(t, "__name__", str(t)))
                if name not in self.messages:
                    self._introspect_and_emit_message(t, name)
                self._seen_types[t] = name
                return name

        # Message (introspect app-specific types)
        if inspect.isclass(t):
            # Don't create placeholder messages for built-in Python types
            if t.__name__ in _SCALAR_NAME_MAP:
                return _SCALAR_NAME_MAP[t.__name__]

            # Check if this is a pure Dict type (like AgentObservations, Observations, ProprioceptiveState)
            if self._is_pure_dict_type(t):
                dict_info = self._get_dict_inheritance_info(t)
                if dict_info:
                    # Always create a message for pure Dict types to avoid nested map syntax
                    wrapper_name = t.__name__
                    if wrapper_name not in self.messages:
                        key_type, value_type = dict_info
                        # Create a message that wraps the map
                        wrapper_fields = [
                            FieldSpec(
                                name="entries",
                                py_type=value_type,
                                optional=False,
                                is_map=True,
                                # Store the key type for proper map rendering
                            )
                        ]
                        # Add key type info for the map field
                        wrapper_fields[0].key_type = key_type
                        self.emit_message(wrapper_name, wrapper_fields)
                    self._seen_types[t] = wrapper_name
                    return wrapper_name

            name = self._make_valid_proto_name(t.__name__)
            if name not in self.messages:
                self._introspect_and_emit_message(t, name)
            self._seen_types[t] = name
            return name

        # Handle typing.Any with smart field name detection
        if t is Any:
            return "bytes"  # Default fallback

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

    def _is_pure_dict_type(self, t: Any) -> bool:
        """Check if a type is a pure Dict type (inherits from Dict but has no additional attributes)."""
        if not inspect.isclass(t):
            return False

        # Check if it inherits from Dict
        if not hasattr(t, "__orig_bases__"):
            return False

        for base in t.__orig_bases__:
            origin = get_origin(base)
            if origin is dict:
                # Check if this class has its own annotated attributes
                return not hasattr(t, "__annotations__") or not any(
                    not field_name.startswith("_")
                    for field_name in t.__annotations__.keys()
                )
        return False

    def _get_dict_inheritance_info(self, t: Any) -> Optional[Tuple[Any, Any]]:
        """Get the key and value types for a Dict inheritance."""
        if not hasattr(t, "__orig_bases__"):
            return None

        for base in t.__orig_bases__:
            origin = get_origin(base)
            if origin is dict:
                args = get_args(base)
                if len(args) == 2:
                    return args[0], args[1]
        return None

    def _smart_type_for_any_field(self, field_name: str) -> str:
        """Return intelligent type mapping for Any fields based on field name."""
        field_lower = field_name.lower()

        # Position fields -> VectorXYZ
        if field_lower in ["position", "pos", "location", "loc", "translation"]:
            return "VectorXYZ"

        # Rotation fields -> QuaternionWXYZ
        if field_lower in ["rotation", "rot", "orientation", "quat", "quaternion"]:
            return "QuaternionWXYZ"

        # Scale fields -> VectorXYZ
        if field_lower in ["scale", "scaling", "size"]:
            return "VectorXYZ"

        # Default fallback
        return "bytes"

    def _type_name_to_field_name(self, type_name: str) -> str:
        """Convert a type name to a snake_case field name."""
        return _type_name_to_field_name_standalone(type_name)

    def _make_valid_proto_name(self, name: str) -> str:
        """Convert a Python type name to a valid protobuf identifier."""
        # Handle common Python type syntax and convert to clean names
        clean_name = name

        # Extract base type name from union types (strip " | None" part)
        # Union types should be handled at field level, not type level
        clean_name = re.sub(r"\s*\|\s*None\s*", "", clean_name)

        # Handle generic types: "tuple[A, B]" -> "TupleAB"
        def replace_generic(match):
            prefix = match.group(1)
            contents = match.group(2)
            # Split on comma and clean each part
            parts = [part.strip() for part in contents.split(",")]
            # Convert each part to PascalCase and join
            clean_parts = []
            for part in parts:
                # Remove spaces and convert to PascalCase
                part_clean = re.sub(r"[^a-zA-Z0-9]", "", part)
                if part_clean:
                    clean_parts.append(part_clean)
            return prefix + "".join(clean_parts)

        clean_name = re.sub(
            r"(tuple|list|dict)\[([^\]]+)\]",
            replace_generic,
            clean_name,
            flags=re.IGNORECASE,
        )

        # Remove all non-alphanumeric characters
        clean_name = re.sub(r"[^a-zA-Z0-9]", "", clean_name)

        # Ensure it starts with a letter and is non-empty
        if not clean_name or clean_name[0].isdigit():
            clean_name = f"Type{clean_name}"

        # Ensure PascalCase (first letter uppercase)
        if clean_name and clean_name[0].islower():
            clean_name = clean_name[0].upper() + clean_name[1:]

        return clean_name or "UnknownType"

    def _resolve_string_type(self, type_name: str) -> Any:
        """Try to resolve a string type name to an actual Python type."""
        # Don't try to resolve generic type expressions like 'tuple[A, B]'
        if re.match(r"(tuple|list|dict)\[", type_name, re.IGNORECASE):
            return None

        # Try to import from the simulator module's namespace and its imports
        try:
            simulator = _load_symbol(SIMULATOR_SYMBOL)
            # Get the module where the simulator is defined
            simulator_module = sys.modules[simulator.__module__]

            # Look for the type in the module's namespace first
            if hasattr(simulator_module, type_name):
                return getattr(simulator_module, type_name)

            # Try to find it in the module's imports by parsing the source
            try:
                source = inspect.getsource(simulator_module)
                tree = ast.parse(source)

                # Find import statements and try to resolve the type
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if hasattr(node, "names"):
                            for alias in node.names:
                                imported_name = alias.name
                                local_name = (
                                    alias.asname if alias.asname else alias.name
                                )

                                if local_name == type_name:
                                    # Found the import, try to load it
                                    try:
                                        mod = importlib.import_module(node.module)
                                        resolved_type = getattr(
                                            mod, imported_name, None
                                        )
                                        if resolved_type is not None:
                                            return resolved_type
                                    except (ImportError, AttributeError):
                                        continue
            except (OSError, SyntaxError):
                pass

            # Also try common modules where these types might be defined
            common_modules = [
                "tbp.monty.frameworks.environments.embodied_environment",
                "tbp.monty.frameworks.actions.actions",
                "tbp.monty.frameworks.models.abstract_monty_classes",
                "tbp.monty.frameworks.models.motor_system_state",
            ]

            for module_name in common_modules:
                try:
                    mod = importlib.import_module(module_name)
                    if hasattr(mod, type_name):
                        return getattr(mod, type_name)
                except ImportError:
                    continue

        except Exception:
            pass

        return None

    def _create_tuple_message(self, tuple_expr: str, message_name: str) -> None:
        """Create a message for tuple expressions like 'tuple[A, B]'."""
        if message_name in self._declared_placeholders:
            return

        # Extract types from tuple expression: "tuple[A, B]" -> ["A", "B"]
        match = re.match(r"tuple\[([^\]]+)\]", tuple_expr, re.IGNORECASE)
        if not match:
            # Fallback to empty message
            body = textwrap.dedent(f"""\
            // Auto-generated message for tuple '{tuple_expr}' (could not parse).
            message {message_name} {{
            }}
            """)
            self.messages[message_name] = body
            self._declared_placeholders.add(message_name)
            return

        # Parse the inner types
        inner_types = [t.strip() for t in match.group(1).split(",")]

        fields = []
        for i, type_name in enumerate(inner_types, 1):
            # Clean up type names (strip union parts like "| None")
            clean_type = re.sub(r"\s*\|\s*None\s*", "", type_name).strip()

            # Create meaningful field name from type name
            field_name = self._type_name_to_field_name(clean_type)

            # Try to resolve and use the type, fallback to bytes for truly unknown types
            resolved = self._resolve_string_type(clean_type)
            if resolved is not None:
                # Use the resolved type - let normal introspection handle it
                proto_type = self.type_to_proto(resolved)
                fields.append(FieldSpec(field_name, resolved, optional=False))
            else:
                # Only fallback to bytes for types we genuinely can't resolve
                fields.append(FieldSpec(field_name, bytes, optional=False))

        if fields:
            # For auto-generated tuple messages, ensure field numbering starts from 1
            # by clearing any existing field numbers for this message
            if message_name in self._field_numbers:
                del self._field_numbers[message_name]
            self.emit_message(message_name, fields)
        else:
            # Fallback to empty message
            body = textwrap.dedent(f"""\
            // Auto-generated message for tuple '{tuple_expr}' (no fields generated).
            message {message_name} {{
            }}
            """)
            self.messages[message_name] = body

        self._declared_placeholders.add(message_name)

    def _introspect_and_emit_message(self, t: Any, name: str) -> None:
        """Introspect a Python type and generate a proper protobuf message."""
        # Avoid redefining messages
        if name in self._declared_placeholders:
            return

        fields = []
        if t is not None:
            fields = self._introspect_type_fields(t)

        if fields:
            self.emit_message(name, fields)
        else:
            # Fallback to empty message
            body = textwrap.dedent(f"""\
            // Auto-generated message for '{name}' (no fields detected).
            message {name} {{
            }}
            """)
            self.messages[name] = body
        self._declared_placeholders.add(name)

    def _introspect_type_fields(self, t: Any) -> List[FieldSpec]:
        """Introspect a Python type and extract its fields for protobuf."""
        fields: List[FieldSpec] = []

        # Handle NewType wrappers
        if hasattr(t, "__supertype__"):
            underlying = t.__supertype__
            return self._introspect_type_fields(underlying)

        # Handle basic tuple types (like VectorXYZ, QuaternionWXYZ)
        origin = get_origin(t)
        if origin in (tuple, Tuple):
            args = get_args(t)
            if args == (float, float, float):
                # VectorXYZ-like
                return [
                    FieldSpec("x", float, optional=False),
                    FieldSpec("y", float, optional=False),
                    FieldSpec("z", float, optional=False),
                ]
            elif args == (float, float, float, float):
                # QuaternionWXYZ-like
                return [
                    FieldSpec("w", float, optional=False),
                    FieldSpec("x", float, optional=False),
                    FieldSpec("y", float, optional=False),
                    FieldSpec("z", float, optional=False),
                ]

        # Handle Protocol classes
        if getattr(t, "_is_protocol", False):
            # Get annotations from the Protocol
            annotations = getattr(t, "__annotations__", {})
            for field_name, field_type in annotations.items():
                # Skip methods and special attributes
                if field_name.startswith("_") or callable(getattr(t, field_name, None)):
                    continue

                is_opt, inner_type = _is_optional(field_type)
                fields.append(
                    FieldSpec(name=field_name, py_type=inner_type, optional=is_opt)
                )
            return fields

        # Handle dataclasses
        if dataclasses.is_dataclass(t):
            for field in dataclasses.fields(t):
                is_opt, inner_type = _is_optional(field.type)

                # Check if this is a dict field for map handling
                origin = get_origin(inner_type)
                is_map_field = origin in (dict, Dict)

                # Smart type detection for Any fields based on field name
                field_type = inner_type
                if inner_type is Any:
                    smart_type_name = self._smart_type_for_any_field(field.name)
                    if smart_type_name != "bytes":
                        # Use the smart type name as a string (will be resolved later)
                        field_type = smart_type_name

                # Improve field naming for ID types
                field_name = field.name
                if (
                    hasattr(inner_type, "__name__")
                    and inner_type.__name__.endswith("ID")
                    and not field.name.endswith("_id")
                ):
                    field_name = f"{field.name}_id"

                fields.append(
                    FieldSpec(
                        name=field_name,
                        py_type=field_type,
                        optional=is_opt or field.default != dataclasses.MISSING,
                        is_map=is_map_field,
                    )
                )
            return fields

        # Handle classes that inherit from Dict[K, V] (like Observations, ProprioceptiveState)
        # Check __orig_bases__ which preserves generic type information
        if hasattr(t, "__orig_bases__"):
            for base in t.__orig_bases__:
                # Check for generic Dict inheritance
                origin = get_origin(base)
                if origin is dict:
                    # This is a class inheriting from Dict[K, V]
                    args = get_args(base)

                    # Check if this class has its own annotated attributes
                    class_fields = []
                    if hasattr(t, "__annotations__"):
                        for field_name, field_type in t.__annotations__.items():
                            if not field_name.startswith("_"):  # Skip private fields
                                is_opt, inner_type = _is_optional(field_type)
                                class_fields.append(
                                    FieldSpec(
                                        name=field_name,
                                        py_type=inner_type,
                                        optional=is_opt,
                                    )
                                )

                    # Ensure referenced types get processed
                    if len(args) == 2:
                        key_type, value_type = args
                        self.type_to_proto(key_type)
                        self.type_to_proto(value_type)

                    # If this class has its own fields, return them (treat as a regular message)
                    # If it has no fields, it's a pure Dict - let the parent handle it as a map
                    return class_fields if class_fields else []

        # Let all types go through normal introspection - no hardcoded special cases

        # If we can't introspect, return empty (will create empty message)
        return []

    def _emit_placeholder_message(self, name: str) -> None:
        # This should now rarely be called since we have introspection
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
                # Check if we have stored key type info
                if hasattr(f, "key_type") and f.key_type:
                    key_proto = self.type_to_proto(f.key_type)
                    value_proto = self.type_to_proto(f.py_type)
                    lines.append(
                        f"  map<{key_proto}, {value_proto}> {f.name} = {field_number};\n"
                    )
                else:
                    # For map fields, we need to determine key and value types
                    origin = get_origin(f.py_type)
                    if origin in (dict, Dict):
                        args = get_args(f.py_type)
                        if len(args) == 2:
                            key_type, value_type = args
                            key_proto = self.type_to_proto(key_type)
                            value_proto = self.type_to_proto(value_type)
                            lines.append(
                                f"  map<{key_proto}, {value_proto}> {f.name} = {field_number};\n"
                            )
                        else:
                            # Fallback
                            lines.append(
                                f"  map<string, {ptype}> {f.name} = {field_number};\n"
                            )
                    else:
                        # Fallback for non-dict map fields
                        lines.append(
                            f"  map<string, {ptype}> {f.name} = {field_number};\n"
                        )
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
        print(
            f"Debug: Parameter {p.name}, type={t}, is_optional={is_opt}, inner={inner}"
        )
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


def _fields_from_return(ret: Any, method_name: str) -> Tuple[str, List[FieldSpec]]:
    """Convert return annotation to response message fields.

    Always returns a specific response message type (never Empty).
    - Treats no annotation OR explicit `-> None` as empty message.
    - For Tuple[...] returns, each element becomes a field:
        * Optional[T] -> optional T
        * literal None -> optional bytes (placeholder to avoid crashes)

    Args:
        ret: Return type annotation
        method_name: Name of the method

    Returns:
        Tuple of (message_name, fields)
    """
    # No return or explicit None -> empty response message
    if ret is inspect._empty or ret is None or ret is type(None):
        return f"{method_name}Response", []

    # Handle string type annotations
    if isinstance(ret, str):
        if ret == "None":
            return f"{method_name}Response", []

        # Check if this is a tuple string like "tuple[A, B]"
        if ret.startswith("tuple[") and ret.endswith("]"):
            # Parse tuple expression: "tuple[A, B]" -> ["A", "B"]
            inner = ret[6:-1]  # Remove "tuple[" and "]"
            type_parts = [t.strip() for t in inner.split(",")]

            fields = []
            for i, type_str in enumerate(type_parts, 1):
                # Clean up type names (strip union parts like "| None")
                clean_type = re.sub(r"\s*\|\s*None\s*", "", type_str).strip()
                is_optional = "| None" in type_str or " | None" in type_str

                # Generate meaningful field name from type
                field_name = _type_name_to_field_name_standalone(clean_type)

                # Improve field naming for ID types
                if clean_type.endswith("ID") and not field_name.endswith("_id"):
                    field_name = f"{field_name}_id"

                fields.append(
                    FieldSpec(
                        name=field_name,
                        py_type=clean_type,
                        optional=is_optional,
                    )
                )

            return f"{method_name}Response", fields

        # For other strings, treat as a single field
        fields = [FieldSpec(name="result", py_type=ret, optional=False)]
        return f"{method_name}Response", fields

    # Optional[...] -> unwrap; Optional[None] -> empty message
    is_opt, inner = _is_optional(ret)
    if is_opt and inner is type(None):
        return f"{method_name}Response", []
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

            # Generate meaningful field name from type
            if hasattr(inner_i, "__name__"):
                field_name = _type_name_to_field_name_standalone(inner_i.__name__)
            elif isinstance(inner_i, str):
                field_name = _type_name_to_field_name_standalone(inner_i)
            else:
                field_name = f"result_{i}"

            fields.append(
                FieldSpec(
                    name=field_name,
                    py_type=inner_i,
                    optional=is_opt_i,
                )
            )
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
        # Request - always create specific request type
        req_fields = _fields_from_parameters(list(sig.parameters.values()))
        req_msg = f"{pascal_method_name}Request"
        builder.emit_message(req_msg, req_fields)

        # Response - always create specific response type
        ret_ann = sig.return_annotation
        resp_msg_name, resp_fields = _fields_from_return(ret_ann, pascal_method_name)
        builder.emit_message(resp_msg_name, resp_fields)
        resp = resp_msg_name

        rpcs.append((pascal_method_name, req_msg, resp))

    # Emit the service
    builder.emit_service("SimulatorService", rpcs)

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
