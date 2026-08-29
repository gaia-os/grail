"""
Simple Elixir actor/critic smoke test for Bayes-style code generation.
"""
import argparse

import numpy as np

from grail.elixir import Elixir, ElixirCritic
from grail.elixir.validator import ElixirInputVar, ElixirOutputVar, ElixirValidator, load_code
from grail.llm.models.deepseek import DeepseekSmall
from grail.llm.models.qwen import Qwen3


class BayesVectorUpdateValidator(ElixirValidator):
    """Validator for a simple Bayesian update over vectors derived from matrix input."""

    is_function = True
    function_name = "bayes_vector_update"
    prompt_description = (
        "Compute a Bayesian posterior over classes using a likelihood matrix, prior vector, "
        "and a one-hot observation vector."
    )
    required_args = (
        ElixirInputVar(
            arg="likelihood_matrix",
            description=(
                "2D numpy array where each row i contains P(observation | class_i) terms "
                "for each possible observation"
            ),
            datatype=np.ndarray,
        ),
        ElixirInputVar(
            arg="prior",
            description="1D numpy array of prior class probabilities that sum to 1",
            datatype=np.ndarray,
        ),
        ElixirInputVar(
            arg="observation_one_hot",
            description="1D one-hot vector selecting the observed outcome",
            datatype=np.ndarray,
        ),
    )
    returned_data = (
        ElixirOutputVar(
            name="posterior",
            description="1D numpy array posterior over classes",
            datatype=np.ndarray,
        ),
        ElixirOutputVar(
            name="updated_alphas",
            description="Dirichlet-like alpha update: alpha + posterior",
            datatype=np.ndarray,
        ),
    )


TASK = """
Write a Python function named bayes_vector_update that:
- Uses numpy.
- Accepts:
  1) likelihood_matrix: shape (n_classes, n_observations)
  2) prior: shape (n_classes,)
  3) observation_one_hot: shape (n_observations,)
- Computes likelihood_vector = likelihood_matrix @ observation_one_hot.
- Computes posterior = prior * likelihood_vector, then normalizes by its sum.
- Creates updated_alphas = ones_like(prior) + posterior.
- Returns (posterior, updated_alphas).
- Handles normalization safely if posterior sum is near zero.
""".strip()


def run_smoke_test(max_iters: int, loop_budget: float) -> int:
    """Run actor/critic generation loop and execute generated function if approved."""
    actor = Qwen3(temperature=0.0, seed=101010)
    # actor = DeepseekSmall(temperature=0.0, seed=101010)
    critic_model = Qwen3(temperature=0.0, seed=101010)
    critic = ElixirCritic(llm=critic_model, task=TASK)
    elixir = Elixir(llm=actor, critic=critic)

    print("Running Elixir critic loop...")
    result = elixir.critic_loop(
        validator=BayesVectorUpdateValidator,
        max_iters=max_iters,
        loop_budget=loop_budget,
        actor_retries=2,
        actor_budget=30,
    )

    print(f"approved: {result['approved']}")
    print(f"iters_used: {result['iters_used']}")

    if not result["approved"] or not result["code"]:
        print("No approved solution was returned.")
        print("Last critic evaluation:")
        print(result["evaluation"])
        return 1

    code = result["code"]
    print("\nApproved code:\n")
    print(code)

    # Compile and execute generated function on a tiny numeric example.
    generated_fn = load_code(code, BayesVectorUpdateValidator.function_name)

    likelihood_matrix = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.6, 0.2, 0.2],
            [0.3, 0.1, 0.6],
        ],
        dtype=float,
    )
    prior = np.array([0.2, 0.5, 0.3], dtype=float)
    observation_one_hot = np.array([0.0, 1.0, 0.0], dtype=float)

    posterior, updated_alphas = generated_fn(likelihood_matrix, prior, observation_one_hot)

    print("\nSanity-check output:")
    print(f"posterior: {posterior}")
    print(f"updated_alphas: {updated_alphas}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple Elixir codegen test with DeepseekSmall actor and Qwen3 critic.",
    )
    parser.add_argument("--max-iters", type=int, default=4, help="Maximum actor/critic loop iterations")
    parser.add_argument("--loop-budget", type=float, default=90.0, help="Total critic loop budget in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_smoke_test(max_iters=args.max_iters, loop_budget=args.loop_budget)


if __name__ == "__main__":
    raise SystemExit(main())


