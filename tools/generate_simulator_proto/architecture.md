Protobuf File Organization Best Practices

1. Keep schemas separate from source code
	•	Why: Protobuf schemas are language-neutral contracts, not tied to Python.
	•	Recommendation: Store them in a top-level proto/ directory rather than inside src/.
	•	Example:

```
repo/
  proto/
    tbp/
      simulator/
        protocol/
          v1/
            protocol.proto
  src/
    tbp/
      monty/
        simulators/
          simulator.py
```

2. Mirror package names with folders
	•	Protobuf package names should match the folder hierarchy.
	•	Package: tbp.simulator.protocol.v1
	•	Folder: proto/tbp/simulator/protocol/v1/

This keeps everything aligned when generating language bindings.

3. Version your packages from the start
	•	Always suffix with .v1 (e.g., tbp.simulator.protocol.v1).
	•	Breaking changes → bump to v2, don’t mutate v1.
	•	This avoids compatibility nightmares once you publish contracts.

4. Generated code should not live next to .proto files
	•	Keep .proto files in proto/.
	•	Generate code into your src/ tree (or a gen/ folder).
	•	Example:

```
src/
  tbp/
    simulator/
      protocol/
        v1/
          protocol_pb2.py
          protocol_pb2.pyi
          protocol_pb2_grpc.py
```

5. Use consistent build tooling
	•	Prefer Buf over raw protoc:
	•	Linting
	•	Breaking change detection
	•	Unified generation config (buf.gen.yaml)

6. Split schemas by domain if they grow large
	•	Start with one umbrella schema (protocol.proto) for the simulator.
	•	As the system grows:

```
proto/tbp/simulator/control/v1/control.proto
proto/tbp/simulator/perception/v1/perception.proto
proto/tbp/simulator/config/v1/config.proto
```

7. Document ownership & intent
	•	Put a README.md in proto/tbp/simulator/ explaining:
	•	What services/messages live here
	•	Versioning policy
	•	Who owns schema evolution

8. Keep generated files out of git (optional but recommended)
	•	Add generated files to .gitignore:

```
src/tbp/simulator/protocol/v1/*_pb2*.py
```

10. Keep comments in sync
	•	Add docstrings in Python → copy into .proto comments.
	•	Well-commented .proto = self-documenting contract for all consumers.

11. Use best practices for Protobuf, including:
	•	Converting snake_case to pascal