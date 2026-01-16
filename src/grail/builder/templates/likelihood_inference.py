import numpy as np
from pydantic import model_validator

from grail.elixir.validator import ElixirValidator, load_code


def indicator_inference(prior: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Update an indicator likelihood prior using Bayesian inference, from N observations of indicator colors

    Args:
        prior: Array of shape (n_category, n_indicator, n_color) for the indicator likelihoods across
            category, indicator, and color.
        observations: Array of shape (N, 1 + n_indicator), where the i-th row observation is n integer
            vector, where j=0 is the category (integer), and j=1:n_indicator is the indicator color (as integer)

    Returns:
        np.ndarray: the updated prior.
    """
    posterior = prior.copy()
    n_cat = prior.shape[0]
    J = prior.shape[1]
    n_color = prior.shape[2]

    for k_val in range(n_cat):
        for j in range(J):  # Iterate over indicators
            for c in range(n_color):  # Iterate over colors
                posterior[k_val, j, c] = prior[k_val, j, c] + np.sum(
                    (observations[:, j + 1] == c) & (observations[:, 0] == k_val)
                )
    epsilon = 1e-10  # Small value to prevent division by zero
    normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
    return normalized


def outer_indicator_inference(data: dict) -> np.ndarray:
    """
    Updates the likelihood matrix based on the observed data.

    Args:
        data: A dictionary containing the observed data for each asset. The keys are the asset names,
            and the values are dictionaries with 'category' and 'indicators' keys. The 'category' key contains
            the class label of the asset, and the 'sampled_colors' key contains a dictionary of the observed
            indicators their their true colors.

    Returns:
        np.ndarray: The likelihood matrix with dimensions (n_category, n_indicator, n_color)
    """
    # Reshape the dictionary input data to a NumPy array
    # N: number of observations, J: number of indicators
    indicators: list[str] = list(data.values())[0]['sampled_colors'].keys()
    categories = ["high_growth", "stable", "time_bomb"]
    n_category = len(categories)
    n_color = 3

    N = len(data)
    J = len(indicators)
    # Each row an asset; Each value is 0,1,2 based on the color of the asset i and indicator j
    # j=0 is category
    observations = np.zeros((N, J + 1), dtype=int)

    for i, (asset_name, asset_data) in enumerate(data.items()):
        # First column is the category
        observations[i, 0] = categories.index(asset_data['category'])
        # Remaining columns are the indicators
        for j, indicator in enumerate(indicators):
            x_ij = asset_data['sampled_colors'][indicator]
            x_ij = 0 if x_ij == 'red' else 1 if x_ij == 'yellow' else 2  # Convert color to integer
            observations[i, j + 1] = x_ij

    # Update beta (likelihood matrix).
    prior = np.ones((n_category, J, n_color))
    posterior = indicator_inference(prior, observations)
    return posterior


#
# def compute_indicator_likelihood_old(data: dict) -> np.ndarray:
#     """
#     Updates the likelihood matrix based on the observed data.
#
#     Args:
#         data: A dictionary containing the observed data for each asset. The keys are the asset names,
#             and the values are dictionaries with 'category' and 'indicators' keys. The 'category' key contains
#             the class label of the asset, and the 'sampled_colors' key contains a dictionary of the observed
#             indicators their their true colors.
#
#     Returns:
#         np.ndarray: The likelihood matrix with dimensions (n_category, n_indicator, n_color)
#     """
#     # Reshape the dictionary input data to a NumPy array
#     # N: number of observations, J: number of indicators
#     indicators: list[str] = list(data.values())[0]['sampled_colors'].keys()
#     categories = ["high_growth", "stable", "time_bomb"]
#     n_category = len(categories)
#     n_color = 3
#
#     N = len(data)
#     J = len(indicators)
#     # Each row an asset; Each value is 0,1,2 based on the color of the asset i and indicator j
#     X = np.zeros((N, J), dtype=int)
#
#     for i, (asset_name, asset_data) in enumerate(data.items()):
#         for j, indicator in enumerate(indicators):
#             x_ij = asset_data['sampled_colors'][indicator]
#             x_ij = 0 if x_ij == 'red' else 1 if x_ij == 'yellow' else 2  # Convert color to integer
#             X[i, j] = x_ij
#
#     # Update beta (likelihood matrix).
#     prior = np.ones((n_category, J, n_color))
#     posterior = prior.copy()
#     for j in range(J):  # Iterate over indicators
#         for k_val, category in enumerate(categories):  # Iterate over categories
#             for c in range(prior.shape[2]):  # Iterate over colors
#                 posterior[k_val, j, c] = prior[k_val, j, c] + np.sum(
#                     (X[:, j] == c) & (np.array([asset['category'] for asset in data.values()]) == category)
#                 )
#     epsilon = 1e-10  # Small value to prevent division by zero
#     normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
#     return normalized
#
#     self.indicators['likelihood'] = (('class', 'indicator', 'color'),
#     posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon))
#
#


class IndicatorLikelihoodInference(ElixirValidator):
    """
    Function for performing inference on a set of data to determine a likelihood matrix for indicators.
    Each indicator has a different set of likelihoods given category.

    Args:
        prior: Array of shape (n_category, n_indicator, n_color) for the indicator likelihoods across
            category, indicator, and color.
        observations: Array of shape (N, 1 + n_indicator), where the i-th row observation is n integer
            vector, where j=0 is the category (integer), and j=1:n_indicator is the indicator color (as integer)

    Returns:
        np.ndarray: the updated prior.
    """
    is_function = True
    function_name = "indicator_likelihood_inference"

    # Ok the prompt here is a bit more involved
    # prompt_description = """
    # Consider the following statistical model and code its solution.
    # We are trying to categorize a company into one of K classes and we observe J categorical indicators,
    # with the same number of categories (colors) C.
    # Each indicator has a different set of likelihoods given classes. We represent the sampling distributions as
    #
    # k ~ Categorical(p)
    # x_j ~ Categorical(A_{j,:,k}) # The array A (J x C x K) contains the likelihoods of each indicator given each of the latent company class k.
    #
    # Update an indicator likelihood prior using Bayesian inference, from N observations of indicator colors and
    # categories. Each indicator has a different set of likelihoods given classes.
    # Tip:
    #     First work out the closed-form Bayesian update for the class probability vector p.
    #     Then extend this to assume the likelihood matrix is also unknown and we must simultaneously infer its parameters.
    #
    # Recall that the unnormalized Bayesian posterior is:
    # posterior = likelihood * prior
    # where the likelihood is the indicator likelihood given the class and the prior is the prior distribution of classes.
    #
    # Your solution may resemble the following:
    #
    # posterior = prior.copy()
    # n_cat = prior.shape[0]
    # J = prior.shape[1]
    # n_color = prior.shape[2]
    #
    # for k_val in range(n_cat):
    #     for j in range(J):  # Iterate over indicators
    #         for c in range(n_color):  # Iterate over colors
    #             posterior[k_val, j, c] = prior[k_val, j, c] ...
    # epsilon = 1e-10  # Small value to prevent division by zero
    # normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
    # return normalized
    # """
    prompt_description = """
    Consider the following statistical model and code its solution. 
    We are trying to categorize a company into one of K classes and we observe J categorical indicators, 
    with the same number of categories (colors) C. 
    Each indicator has a different set of likelihoods given classes. We represent the sampling distributions as

    k ~ Categorical(p)
    x_j ~ Categorical(A_{j,:,k}) # The array A (J x C x K) contains the likelihoods of each indicator given each of the latent company class k.

    Update an indicator likelihood prior using Bayesian inference, from N observations of indicator colors and 
    categories. Each indicator has a different set of likelihoods given classes.
    """

    required_args = tuple(
        [
            {
                "arg": "prior",
                "description": "Prior matrix",
                "datatype": np.ndarray
            },
            {
                "arg": "observations",
                "description": "Observations matrix",
                "datatype": np.ndarray
            },
        ]
    )
    returned_data = tuple(
        [
            {
                "name": "Posterior",
                "description": "The bayesian posterior",
                "type": np.ndarray
            }
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def test_output(self):
        """
        Verify some aspects of the return value
        """
        # Create a dummy prior and observations
        prior = np.ones((3, 2, 3)) / 3
        observations = np.array(
            [
                [0, 1, 2],
                [1, 2, 2],
                [2, 1, 0]
            ]
        )

        # Call the function
        func = load_code(self.code, self.function_name)
        posterior = func(prior, observations)

        assert posterior.shape == (3, 2, 3), "Output shape is incorrect"
        assert np.allclose(posterior.sum(axis=2), 1), "Posterior probabilities do not sum to 1"
        assert not np.allclose(posterior, prior), "Function is identity operation"

        return self


# For testing
solution_string = '''
import numpy as np


def indicator_likelihood_inference(prior: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Update an indicator likelihood prior using Bayesian inference, from N observations of indicator colors

    Args:
        prior: Array of shape (n_category, n_indicator, n_color) for the indicator likelihoods across
            category, indicator, and color.
        observations: Array of shape (N, 1 + n_indicator), where the i-th row observation is n integer
            vector, where j=0 is the category (integer), and j=1:n_indicator is the indicator color (as integer)

    Returns:
        np.ndarray: the updated prior.
    """
    posterior = prior.copy()
    n_cat = prior.shape[0]
    J = prior.shape[1]
    n_color = prior.shape[2]

    for k_val in range(n_cat):
        for j in range(J):  # Iterate over indicators
            for c in range(n_color):  # Iterate over colors
                posterior[k_val, j, c] = prior[k_val, j, c] + np.sum(
                    (observations[:, j + 1] == c) & (observations[:, 0] == k_val)
                )
    epsilon = 1e-10  # Small value to prevent division by zero
    normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
    return normalized
'''

TASK_HARD = """
We are trying to categorize a company into one of K classes and we observe J categorical indicators.  
Each indicator can take one of C colors, that characterize the company's value in the dimension of that indicator.
Each indicator has a different set of likelihoods given classes. We represent the sampling distributions as

k ~ Categorical(p)
x_j ~ Categorical(A_{j,:,k}) # The array A (J x C x K) contains the likelihoods of each indicator given each of the latent company class k.

The goal is to update an indicator likelihood prior using Bayesian inference, from N observations of indicator colors and 
categories. Each indicator has a different set of likelihoods given classes.
"""

TASK_MEDIUM = TASK_HARD + """

Tip:
    First work out the closed-form Bayesian update for the class probability vector p.
    Then extend this to assume the likelihood matrix is also unknown and we must simultaneously infer its parameters.
"""

TASK_EASY = TASK_MEDIUM + """

Recall that the unnormalized Bayesian posterior is:
posterior = likelihood * prior
where the likelihood is the indicator likelihood given the class and the prior is the prior distribution of classes.

Your solution may resemble the following:

posterior = prior.copy()
n_cat = prior.shape[0]
J = prior.shape[1]
n_color = prior.shape[2]

for k_val in range(n_cat):
    for j in range(J):  # Iterate over indicators
        for c in range(n_color):  # Iterate over colors
            posterior[k_val, j, c] = prior[k_val, j, c] ...
epsilon = 1e-10  # Small value to prevent division by zero
normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
return normalized
"""
