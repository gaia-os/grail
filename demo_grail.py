import torch
from grail.frame.builder import Builder
from grail.engine import Engine
from grail.runner import Runner

def main():
    print("Initializing GRAIL...")
    builder = Builder()
    
    # 1. Create a Frame
    frame = builder.new_frame("HealthModel")
    
    # 2. Add Variables
    # Exercise: Bernoulli(0.5)
    # We use a helper from Frame, or manually add.
    exercise_id = frame.add_variable(
        name="Exercise", 
        dist="Bernoulli", 
        params={"probs": 0.5}
    )
    
    # Health: Normal(loc=Exercise, scale=0.1)
    # Note: In this v0.X MVP, passing the ID string means we use the value of that parent node
    # So if Exercise=1, Health ~ Normal(1, 0.1). If Exercise=0, Health ~ Normal(0, 0.1).
    health_id = frame.add_variable(
        name="Health",
        dist="Normal",
        params={"loc": exercise_id, "scale": 0.1}
    )
    
    # 3. Add Dependency (Causal Link)
    frame.add_dependency(exercise_id, health_id)
    
    print(f"Constructed Frame: {frame.name}")
    print(f"Nodes: {len(frame.graph.graph.nodes)}")
    
    # 4. Build Engine
    print("Building Engine...")
    engine = Engine(frame)
    model = engine.get_model()
    
    # 5. Run Simulation
    print("Running Simulation (Prior Predictive)...")
    runner = Runner(model)
    samples = runner.simulate(num_samples=5)
    
    print("Simulation Results:")
    for k, v in samples.items():
        print(f"{k}: {v}")

    # 6. Run Intervention (Do-Operator)
    # Force Exercise to be 1 (True)
    print("\nIntervention: do(Exercise=1.0)")
    # Note: pyro.do expects the site name (which is 'name' in add_variable), not ID
    # My Engine uses node.name for pyro.sample().
    do_result = runner.do_operation({"Exercise": torch.tensor(1.0)}, num_samples=5)
    
    print("Intervention Results (Health should be around 1.0):")
    for k, v in do_result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
