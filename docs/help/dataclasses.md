# Python Dataclasses Quick Guide

Dataclasses are a standard-library feature (`dataclasses`) for building classes that mainly store data, while avoiding boilerplate like hand-written constructors and repr methods.

In practice, they are used **instead of** writing a regular class with lots of repetitive 
"plumbing" (`__init__`, `__repr__`, `__eq__`, and default handling), or instead of using lighter but more 
limited shapes like tuples/dicts when you want named fields plus methods.

They solve a specific problem: many model-like classes spend more code on setup and maintenance boilerplate 
than on domain logic. `@dataclass` removes that friction, keeps intent clear at the field level, 
and makes defaults (especially mutable defaults via `default_factory`) safer and more explicit.

## Why use dataclasses?

- Less boilerplate for data-centric classes.
- Readable field declarations with type hints.
- Sensible generated methods by default.
- Easy handling of defaults, especially mutable defaults.

## Minimal example

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
```

This is roughly equivalent to manually writing:

- `__init__(self, x, y)`
- `__repr__(self)`
- `__eq__(self, other)`

## Dataclass behavior in this project (`Variable`)

From `src/grail/frame/variable.py`:

```python
@dataclass
class Variable:
    name: str
    prior: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    code: str | None = None
    observations: Any | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
```

### What is auto-generated?

By default, `@dataclass` generates:

- `__init__` with parameters in field order.
- `__repr__` showing field values.
- `__eq__` comparing instances by field values.

So you can do:

```python
v = Variable(name="blood_pressure")
print(v)  # readable repr
```

### Why `default_factory` matters

`prior` and `attributes` are dictionaries (mutable objects). Use:

```python
field(default_factory=dict)
```

instead of:

```python
prior: dict[str, Any] = {}
```

Reason: mutable defaults would otherwise be shared across all instances.

## Common dataclass options

```python
@dataclass(
    frozen=False,   # if True, instances are immutable-like
    order=False,    # if True, adds ordering methods (<, <=, >, >=)
    slots=False,    # if True, reduces memory and can speed attribute access
    kw_only=False,  # if True, fields become keyword-only in __init__
)
class Example:
    value: int
```

Notes:

- `frozen=True` prevents assignment after construction (`obj.x = 3` fails).
- `order=True` requires meaningful field ordering semantics.
- With `order=True`, comparisons between instances are tuple-like and lexicographic by field declaration order. For example, if fields are `(a, b, c)`, then `x < y` compares `a` first, then `b`, then `c`.
- `order=True` is best when your instances have a natural total ordering (for example, if it has fields `(timestamp, id)`). Avoid it when ordering is ambiguous or domain-specific rules are more complex than field order.
- `order=True` works with generated equality semantics; if you need custom comparison behavior, prefer implementing comparison methods manually.
- `slots=True` can help performance in high-volume object creation.

## Derived/computed fields with `init=False`

Use `init=False` for fields not passed by caller:

```python
from dataclasses import dataclass, field

@dataclass
class Token:
    raw: str
    normalized: str = field(init=False)

    def __post_init__(self) -> None:
        self.normalized = self.raw.strip().lower()
```

## Post-initialization with `__post_init__`

`__post_init__` runs after auto-generated `__init__`. Use it for:

- Validation.
- Computed fields.
- Cross-field consistency checks.

Example:

```python
from dataclasses import dataclass

@dataclass
class Probability:
    p: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p must be between 0 and 1")
```

## Safe patterns and gotchas

- Use `field(default_factory=...)` for mutable defaults (`list`, `dict`, `set`).
- Keep field order valid: non-default fields must come before default fields.
- Type hints are not runtime enforcement; use validation if needed.
- Prefer plain classes when behavior dominates over stored data.

## Serialization helpers

`dataclasses` provides utility functions:

```python
from dataclasses import asdict, astuple

# asdict(obj)  -> deep-converted dict
# astuple(obj) -> tuple
```

These are convenient for logging, debug snapshots, or simple serialization.

## Practical usage tips for Grail

- Keep model-like containers (such as `Variable`) as dataclasses.
- Place domain operations as methods on the dataclass (already done in `Variable`).
- Use `__post_init__` if later you want to validate distribution schema shape.
- Consider `slots=True` if profiling shows many `Variable` allocations.
