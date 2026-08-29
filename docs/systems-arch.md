# Systems Architecture

This document is specifically about the **tooling and storage stack** around GRAIL: config formats, metadata storage, run outputs, and analysis tools.
It is not the high-level model architecture doc.

## Current / MVP Direction

### 1. Frame definitions: YAML

Frames should be defined in **YAML**.

Why:

- easier to read and edit than JSON
- better for nested model / graph configuration
- a natural fit for Interpreter-produced config files
- easy to version in git

Recommended tooling:

- `PyYAML` for a simple start
- `ruamel.yaml` later if we want better round-tripping / comment preservation

### 2. Schema validation and runtime models: Pydantic

Frame YAML should load into **Pydantic** models.

Why:

- validates frame specs on load
- gives typed Python objects after parsing
- supports defaults and optional fields cleanly
- makes schema evolution easier with explicit `version` fields and migration logic

This keeps YAML as the source-of-truth format while making runtime usage ergonomic.

### 3. Metadata database: SQLite

For project metadata, use **SQLite** first.

This database should track things like:

- registered Frames
- variables / model metadata
- generated operation records
- runs
- run status
- artifact paths
- timestamps

Why SQLite:

- zero setup
- local-first
- easy to ship with the project
- enough for MVP and single-user workflows

### 4. Avoid hand-writing SQL: SQLModel

The user-facing persistence layer should be **Python-first**, not raw SQL-first.

Recommended tooling:

- `SQLModel` as the preferred ORM-ish layer
- `SQLAlchemy` underneath via `SQLModel`

Why:

- avoids manual SQL for normal development work
- still gives structured relational metadata
- makes it easy to swap SQLite for PostgreSQL later if needed

So the intended experience is more like:

- define Python models
- query with Python APIs
- keep raw SQL optional and rare

### 5. Run outputs and analysis data: Parquet

Simulation results should generally be stored as **Parquet** files rather than packed into the metadata database.

Examples:

- posterior samples
- prior predictive samples
- posterior predictive draws
- trajectories
- summaries
- diagnostics

Why Parquet:

- efficient for tabular / columnar outputs
- good for larger result sets
- works well with multiple analysis tools
- keeps the relational DB from becoming a blob store

Recommended tooling:

- `pyarrow`

### 6. Primary analysis interface: Polars

For most analysis work, prefer **Polars** as the main Python-facing tool.

Why:

- fast dataframe operations
- cleaner than forcing SQL everywhere
- a good fit for Parquet-backed workflows

This lets analysis stay Pythonic for common tasks.

### 7. Heavy local analysis engine: DuckDB

Use **DuckDB** as an optional but very strong local analytics engine.

Best use cases:

- querying many Parquet result files together
- comparing runs at scale
- batch summaries and aggregate analysis
- ad hoc exploration once result volumes grow

Important framing:

- DuckDB is a power tool, not the default user interface
- it should often sit behind helper functions or analysis utilities
- users should not be forced into writing SQL for ordinary workflows

---

## Recommended MVP Stack

### Config and model spec

- YAML
- `PyYAML`
- `Pydantic`

### Metadata and registry

- SQLite
- `SQLModel`

### Outputs and analysis

- Parquet
- `pyarrow`
- `Polars`
- optional `DuckDB`

### Existing scientific / graph layer this sits around

- `numpy`
- `pandas` where still convenient
- `networkx`
- `pyro-ppl`

---

## Production Upgrades

These are the larger-scale upgrades to consider once the MVP stack starts to strain.

### 1. Metadata DB upgrade: PostgreSQL

Move from SQLite to **PostgreSQL** when you need:

- concurrent access
- multiple users or services
- more formal migrations
- hosted / managed DB infrastructure

Likely added tooling:

- `psycopg`
- `alembic`

### 2. Remote artifact and result storage

Keep Parquet for outputs, but move storage from local disk to object storage when needed.

Likely targets:

- S3
- MinIO
- GCS
- Azure Blob Storage

Useful tooling:

- `fsspec`

This keeps large artifacts out of the metadata database.

### 3. Keep DuckDB as an analytics backend

DuckDB can remain useful even in a bigger system.

Production-like uses:

- scheduled analysis over many result files
- run comparison utilities
- reproducible reports
- batch sensitivity analysis

### 4. Job / workflow infrastructure

Only add this once runs become operationally heavy.

Possible tools:

- `Celery`
- `Dramatiq`
- `Prefect`
- `Dagster`

Use these when runs need to be:

- queued
- retried
- parallelized
- executed remotely

### 5. Optional specialized databases

These are not part of the core recommendation, but could matter later.

#### Graph databases

Examples:

- Neo4j
- Memgraph

Only worth it if graph traversal / relationship queries become much more central than the current in-memory graph workflow.

#### Vector databases

Examples:

- Qdrant
- Weaviate
- `pgvector` on PostgreSQL

Only worth it if retrieval over documents / research artifacts becomes a major subsystem.

---

## Practical Summary

If we stay disciplined, the most sensible current stack is:

1. **YAML** for Frame specs
2. **Pydantic** for validation and runtime models
3. **SQLite + SQLModel** for metadata and run registries
4. **Parquet + PyArrow** for run outputs
5. **Polars** for normal analysis
6. **DuckDB** for heavier local analytics

That gives GRAIL a strong local-first architecture without forcing raw SQL or heavyweight infrastructure too early.

