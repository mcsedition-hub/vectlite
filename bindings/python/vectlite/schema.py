"""Optional typed schema validation for vectlite metadata.

Define a schema to get clear error messages when metadata doesn't match
expected types::

    from vectlite import schema

    db = vectlite.open("my.vdb", dimension=384)

    # Define the schema
    s = schema.Schema({
        "price": "number",
        "title": "string",
        "tags": "array<string>",
        "published": "boolean",
        "author": {
            "name": "string",
            "age": "number",
        },
    })

    # Attach to a database (stores in .vdb.schema.json sidecar)
    s.save(db)

    # Load existing schema
    s = schema.load(db)

    # Validate metadata before writing
    s.validate({"price": 9.99, "title": "Hello"})         # OK
    s.validate({"price": "not a number"})                  # raises SchemaError

    # Validated wrapper methods
    validated = schema.validated(db, s)
    validated.upsert("id", vector, {"price": 9.99})        # validates then writes
"""

from __future__ import annotations

import json
import os
from typing import Any, TYPE_CHECKING

from ._vectlite import VectLiteError

if TYPE_CHECKING:
    from . import Database

# Supported scalar types
_SCALAR_TYPES = {"string", "number", "integer", "boolean", "null", "any"}


class SchemaError(VectLiteError):
    """Raised when metadata does not match the defined schema."""

    pass


class Schema:
    """A typed metadata schema for a vectlite database.

    Schema definitions map field names to type specifiers:

    - ``"string"`` -- ``str``
    - ``"number"`` -- ``int`` or ``float``
    - ``"integer"`` -- ``int`` only
    - ``"boolean"`` -- ``bool``
    - ``"null"`` -- ``None``
    - ``"any"`` -- any value (no type check)
    - ``"array"`` -- ``list`` of any values
    - ``"array<string>"`` -- ``list[str]``
    - ``"array<number>"`` -- ``list[int | float]``
    - ``"object"`` -- ``dict``
    - ``{"field": "type", ...}`` -- nested object with its own schema

    Fields not listed in the schema are allowed (open schema).  Use
    ``strict=True`` to reject unknown fields.
    """

    def __init__(
        self,
        fields: dict[str, Any],
        *,
        strict: bool = False,
    ) -> None:
        self.fields = fields
        self.strict = strict
        self._compiled = _compile_schema(fields)

    def validate(self, metadata: dict[str, Any] | None) -> None:
        """Validate metadata against the schema.

        Raises :class:`SchemaError` with a descriptive message on mismatch.
        Does nothing if metadata is ``None``.
        """
        if metadata is None:
            return
        _validate(metadata, self._compiled, self.strict, path="")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the schema to a JSON-safe dict."""
        return {"fields": self.fields, "strict": self.strict}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Schema:
        """Deserialize a schema from a dict."""
        return cls(data["fields"], strict=data.get("strict", False))

    def save(self, db: Database) -> None:
        """Persist the schema in a sidecar ``.schema.json`` file next to the DB."""
        schema_path = _schema_path(db)
        with open(schema_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        return f"Schema(fields={self.fields!r}, strict={self.strict})"


def load(db: Database) -> Schema | None:
    """Load a schema from the sidecar file, or return ``None`` if none exists."""
    schema_path = _schema_path(db)
    if not os.path.isfile(schema_path):
        return None
    with open(schema_path, "r") as f:
        data = json.load(f)
    return Schema.from_dict(data)


def _schema_path(db: Database) -> str:
    return db.path + ".schema.json"


# -------------------------------------------------------------------
# Type checking internals
# -------------------------------------------------------------------

_CompiledField = tuple[str, Any]  # (type_name, extra)


def _compile_schema(fields: dict[str, Any]) -> dict[str, _CompiledField]:
    compiled: dict[str, _CompiledField] = {}
    for key, type_spec in fields.items():
        if isinstance(type_spec, dict):
            # Nested object schema
            compiled[key] = ("object_schema", _compile_schema(type_spec))
        elif isinstance(type_spec, str):
            compiled[key] = _parse_type_str(type_spec)
        else:
            raise SchemaError(f"Invalid type specifier for field '{key}': {type_spec!r}")
    return compiled


def _parse_type_str(spec: str) -> _CompiledField:
    spec = spec.strip().lower()
    if spec in _SCALAR_TYPES:
        return (spec, None)
    if spec == "array":
        return ("array", None)
    if spec == "object":
        return ("object", None)
    if spec.startswith("array<") and spec.endswith(">"):
        inner = spec[6:-1].strip()
        if inner not in _SCALAR_TYPES:
            raise SchemaError(f"Unsupported array element type: '{inner}'")
        return ("typed_array", inner)
    raise SchemaError(f"Unknown type specifier: '{spec}'")


def _check_type(value: Any, type_name: str) -> bool:
    if type_name == "any":
        return True
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return False


def _validate(
    metadata: dict[str, Any],
    compiled: dict[str, _CompiledField],
    strict: bool,
    path: str,
) -> None:
    if not isinstance(metadata, dict):
        raise SchemaError(f"{'metadata' if not path else path}: expected an object, got {type(metadata).__name__}")

    if strict:
        extra_keys = set(metadata.keys()) - set(compiled.keys())
        if extra_keys:
            raise SchemaError(
                f"{path or 'metadata'}: unknown field(s): {', '.join(sorted(extra_keys))}"
            )

    for key, (type_name, extra) in compiled.items():
        if key not in metadata:
            continue  # missing fields are allowed (use a separate 'required' check if needed)

        value = metadata[key]
        field_path = f"{path}.{key}" if path else key

        if value is None:
            # None is always allowed (represents missing / null)
            continue

        if type_name in _SCALAR_TYPES:
            if not _check_type(value, type_name):
                expected = type_name
                got = type(value).__name__
                raise SchemaError(f"Field '{field_path}': expected {expected}, got {got} ({value!r})")

        elif type_name == "array":
            if not isinstance(value, list):
                raise SchemaError(f"Field '{field_path}': expected array, got {type(value).__name__}")

        elif type_name == "typed_array":
            if not isinstance(value, list):
                raise SchemaError(f"Field '{field_path}': expected array<{extra}>, got {type(value).__name__}")
            for i, item in enumerate(value):
                if not _check_type(item, extra):
                    raise SchemaError(
                        f"Field '{field_path}[{i}]': expected {extra}, got {type(item).__name__} ({item!r})"
                    )

        elif type_name == "object":
            if not isinstance(value, dict):
                raise SchemaError(f"Field '{field_path}': expected object, got {type(value).__name__}")

        elif type_name == "object_schema":
            if not isinstance(value, dict):
                raise SchemaError(f"Field '{field_path}': expected object, got {type(value).__name__}")
            _validate(value, extra, strict, field_path)


class ValidatedDatabase:
    """Thin wrapper around a :class:`vectlite.Database` that validates
    metadata on every write operation."""

    def __init__(self, db: Database, schema: Schema) -> None:
        self._db = db
        self._schema = schema

    @property
    def db(self) -> Database:
        return self._db

    @property
    def schema(self) -> Schema:
        return self._schema

    def upsert(self, id: str, vector: list[float], metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._schema.validate(metadata)
        self._db.upsert(id, vector, metadata, **kwargs)

    def insert(self, id: str, vector: list[float], metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._schema.validate(metadata)
        self._db.insert(id, vector, metadata, **kwargs)

    def update_metadata(self, id: str, metadata: dict[str, Any], **kwargs: Any) -> bool:
        self._schema.validate(metadata)
        return self._db.update_metadata(id, metadata, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)


def validated(db: Database, schema: Schema) -> ValidatedDatabase:
    """Create a validated wrapper around a database."""
    return ValidatedDatabase(db, schema)


__all__ = [
    "Schema",
    "SchemaError",
    "ValidatedDatabase",
    "load",
    "validated",
]
