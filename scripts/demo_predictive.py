import torch
from grail.engine import Engine
from grail.frame import FrameRepository
from grail.runner import Runner


def main():
    print("Initializing GRAIL...")
    repository = FrameRepository()
    model_file = "examples/health_model.yaml"

    # 1. Load and validate YAML spec, then compile to runtime Frame.
    spec_path = repository.path_for("examples/health_model")
    spec = repository.load_spec(model_file)
    frame = repository.load(model_file)

    print(f"Loaded Spec: {spec_path}")
    print(f"Variables: {len(spec.variables)}, Dependencies: {len(spec.dependencies)}")
    print(f"Constructed Frame: {frame.name}")
    print(f"Nodes: {len(frame.graph.graph.nodes)}")

    # 2. Build Engine
    print("Building Engine...")
    engine = Engine(frame)
    model = engine.get_model()

    # 3. Run Simulation
    print("Running Simulation (Prior Predictive)...")
    runner = Runner(model)
    samples = runner.simulate(num_samples=5)

    print("Simulation Results:")
    for k, v in samples.items():
        print(f"{k}: {v}")

    # 4. Run Intervention (Do-Operator)
    # Force Exercise to be 1 (True)
    print("\nIntervention: do(Exercise=1.0)")
    do_result = runner.do_operation({"Exercise": torch.tensor(1.0)}, num_samples=5)

    print("Intervention Results (Health should be around 1.0):")
    for k, v in do_result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
