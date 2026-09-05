"""Run a complete toy workflow on the health example Frame."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from grail.engine import Engine
from grail.frame import FrameRepository
from grail.frame.registry import FrameRegistry
from grail.inference import BetaBernoulliInference
from grail.runner import Runner
from grail.runner.utils import list_runs
from grail.settings import PROJECT_ROOT


def _tensor_mean(value: Any) -> float | None:
    """Return a numeric mean for tensor-like outputs when available."""
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return None


def run(samples: int) -> int:
    repository = FrameRepository()
    frame = repository.load("examples/health_model.yaml")

    spec_path = os.path.join(repository.root, "examples", "health_model.yaml")
    FrameRegistry().register(frame.to_spec(), spec_path)

    engine = Engine(frame)
    runner = Runner(engine.get_model(), frame=frame)

    print("=== Health Toy Workflow ===")
    print(f"Frame: {frame.name}")
    print(f"Definition hash: {frame.definition_hash}")

    print("\n1) Prior predictive simulation")
    prior = runner.simulate(num_samples=samples)
    exercise_mean = _tensor_mean(prior.get("Exercise"))
    health_mean = _tensor_mean(prior.get("Health"))
    print(f"Exercise mean over {samples} draws: {exercise_mean}")
    print(f"Health mean over {samples} draws: {health_mean}")

    observations_file = os.path.join(PROJECT_ROOT, "data", "observations", "examples", "health_model.json")
    print("\n2) Load example observations")
    loaded_batch_ids = frame.load_observations(observations_file)
    print(f"Loaded batches: {loaded_batch_ids}")

    print("\n3) Exact Beta-Bernoulli inference")
    posteriors = runner.infer(BetaBernoulliInference())
    posterior = posteriors.get("ExerciseRate")
    if posterior is None:
        print("No posterior was produced for ExerciseRate.")
    else:
        print(
            "ExerciseRate posterior:",
            {
                "distribution": posterior.distribution,
                "params": posterior.params,
                "metadata": posterior.metadata,
            },
        )

    print("\n4) Inspect durable run record")
    runs = list_runs(frame, strategy_id=BetaBernoulliInference.name)
    if not runs:
        print("No run records found.")
        return 1

    latest = runs[0]
    print(f"Latest run id: {latest.id}")
    print(f"Status: {latest.status.value}")
    print(f"Artifacts: {latest.artifact_paths}")

    metadata_path = Path(latest.artifact_paths["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print("Run metadata summary:")
    print(
        json.dumps(
            {
                "id": metadata["id"],
                "strategy_id": metadata["strategy_id"],
                "status": metadata["status"],
                "observation_batch_ids": metadata["observation_batch_ids"],
                "completed_at": metadata["completed_at"],
            },
            indent=2,
            sort_keys=True,
        )
    )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the health toy model workflow.")
    parser.add_argument("--samples", type=int, default=200, help="Number of prior predictive draws")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    return run(samples=args.samples)


if __name__ == "__main__":
    raise SystemExit(main())
