import torch
from grail.builder import Builder
from grail.engine import Engine
from grail.runner import Runner

def test_full_simulation_flow():
    # 1. Builder & Frame
    builder = Builder()
    frame = builder.new_frame("TestModel")
    
    assert frame.name == "TestModel"
    assert len(frame.graph.graph.nodes) == 0

    # 2. Add Variables
    # A -> B
    # A ~ Bernoulli(0.5)
    id_a = frame.add_variable("A", "Bernoulli", {"probs": 0.5})
    
    # B ~ Normal(A, 0.1)
    id_b = frame.add_variable("B", "Normal", {"loc": id_a, "scale": 0.1})
    
    frame.add_dependency(id_a, id_b)
    
    assert len(frame.graph.graph.nodes) == 2
    assert len(frame.graph.graph.edges) == 1
    
    # 3. Engine Construction
    engine = Engine(frame)
    model = engine.get_model()
    
    assert callable(model)
    
    # 4. Runner & Simulation
    runner = Runner(model)
    
    # Forward simulation
    samples = runner.simulate(num_samples=100)
    
    assert "A" in samples
    assert "B" in samples
    assert samples["A"].shape == (100,)
    
    # Check simple logic: Values of B should be close to A (0 or 1)
    # If A=0, B~N(0, 0.1). If A=1, B~N(1, 0.1).
    # Mean of B when A=0 should be approx 0.
    
    mask_0 = (samples["A"] == 0.0)
    if mask_0.sum() > 10:
        b_given_0 = samples["B"][mask_0]
        assert torch.abs(b_given_0.mean()) < 0.2

    mask_1 = (samples["A"] == 1.0)
    if mask_1.sum() > 10:
        b_given_1 = samples["B"][mask_1]
        assert torch.abs(b_given_1.mean() - 1.0) < 0.2

def test_intervention():
    builder = Builder()
    frame = builder.new_frame("CausalTest")
    
    # X -> Y
    id_x = frame.add_variable("X", "Normal", {"loc": 0.0, "scale": 1.0})
    id_y = frame.add_variable("Y", "Normal", {"loc": id_x, "scale": 0.1})
    
    frame.add_dependency(id_x, id_y)
    
    engine = Engine(frame)
    runner = Runner(engine.get_model())
    
    # do(X=10.0) -> Y should be around 10.0
    intervention = {"X": torch.tensor(10.0)}
    results = runner.do_operation(intervention, num_samples=50)
    
    assert "Y" in results
    assert torch.abs(results["Y"].mean() - 10.0) < 0.5
