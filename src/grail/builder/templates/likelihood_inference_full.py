import numpy as np
from grail.synthetic import get_ground_truth_samples, Indicators
from pydantic import model_validator

from grail.elixir.validator import ElixirException, ElixirValidator, load_code


def _format_data(data: dict) -> dict:
    """
    Format the raw data dictionary for input to the likelihood function
    """
    # Convert the data dictionary to a format suitable for the likelihood function
    formatted_data = {}
    for asset_name, asset_data in data.items():
        category = asset_data['category']
        sampled_colors = asset_data['sampled_colors']
        formatted_data[asset_name] = {
            'category': category,
            'sampled_colors': sampled_colors
        }
    return formatted_data


def get_mappings():
    """
    Get the mappings for the likelihood function
    """
    color_mapping = {
        'red': 0,
        'yellow': 1,
        'green': 2
    }
    indicator_mapping = {
        name: i for i, name in enumerate(Indicators.names)
    }
    category_mapping = {
        'high_growth': 0,
        'stable': 1,
        'time_bomb': 2
    }
    return color_mapping, indicator_mapping, category_mapping


class LikelihoodInferenceFull(ElixirValidator):
    """
    Full formulation for performing inference provided only the dictionary of asset data observations to build
    the indicator likelihood matrix. Each indicator has a different set of likelihoods given category.

    Mapping keys can be used to extract values in data dict.
    Args:
        data: A dictionary containing the observed data for each asset, with the following format:
            {
                <asset_1>: {
                    "category": <cat>,
                    "sampled_colors": {
                        <ind_1>: <color>,
                        <ind_2>: <color>,
                        ...
                },
                <asset_2>: {...},
                ...
            }
            The sampled indicator color strings correspond to the assets performance in that indicator dimension.
            The allowed string values for category, indicator, and color can be retrieved from the
            mapping arguments provided.
        color_mapping: Mapping of color names to likelihood array index.
        indicator_mapping: Mapping of indicator names to likelihood array index.
        category_mapping: Mapping of category names to likelihood array index.
    Returns:
        np.ndarray: The likelihood matrix with dimensions (n_category, n_indicator, n_color)
    """
    is_function = True
    function_name = "likelihood_inference_full"

    prompt_description = """
    Consider the following statistical model and code its solution. 
    We are trying to categorize a company into one of K classes and we observe J categorical indicators, 
    with the same number of categories (colors) C. 
    Each indicator has a different set of likelihoods given classes. We represent the sampling distributions as

    k ~ Categorical(p)
    x_j ~ Categorical(A_{j,:,k}) # The array A (J x C x K) contains the likelihoods of each indicator given each of the latent company class k.

    From a data dictionary of asset category and indicator color observations, use Bayesian inference to
    return the indicator likelihood matrix A.
    
    Here is a description of the input data dictionary:
    "The keys are the asset names, and the values include dictionaries with 'category' and 'sampled_colors' keys. 
    Each 'category' item contains the category label of the asset, and the 'sampled_colors' item is a dictionary 
    keyed by indicator name, and observed color (str). Colors are one of three ['red', 'yellow', 'green'], 
    corresponding to the asset's performance in this indicator dimension."
    """

    required_args = tuple(
        [
            {
                "arg": "data",
                "description": "Data dictionary of asset observations",
                "datatype": dict
            },
            {
                "arg": "color_mapping",
                "description": "Mapping of color names to likelihood array index.",
                "datatype": dict
            },
            {
                "arg": "indicator_mapping",
                "description": "Mapping of indicator names to likelihood array index.",
                "datatype": dict
            },
            {
                "arg": "category_mapping",
                "description": "Mapping of category names to likelihood array index.",
                "datatype": dict
            }
        ]
    )
    returned_data = tuple(
        [
            {
                "name": "Likelihood Matrix",
                "description": "The likelihood matrix with dimensions (n_category, n_indicator, n_color)",
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

        We can compare output values with the function above, indicator_likelihood()
        """
        # Load some sample training data
        n_shots, train_size = 5, 100  # Produces 15 samples
        training_data = get_ground_truth_samples(n_shots, train_size)
        # Get likelihood solution
        # Reformat the data
        training_data = _format_data(training_data)
        color_mapping, indicator_mapping, category_mapping = get_mappings()
        solution = likelihood_inference_soln(training_data, color_mapping, indicator_mapping, category_mapping)

        # Get solution from the function
        func = load_code(self.code, self.function_name)
        try:
            result = func(training_data, color_mapping, indicator_mapping, category_mapping)
        except Exception as e:
            raise ElixirException("Exception encountered running the function", e, code=self.code)

        # Compare result with solution
        if result.shape != solution.shape:
            raise ElixirException(
                f"Result shape {result.shape} does not match solution shape {solution.shape}",
                code=self.code
            )

        if not np.allclose(result.sum(axis=2), 1):
            raise ElixirException("Posterior probabilities do not sum to 1", code=self.code)

        # Relative tolerance
        eps = 1e-6
        if not np.allclose(result, solution, rtol=eps):
            raise ElixirException(
                f"Result does not match the expected solution.\nresult[0] = {result[0]}",
                code=self.code
            )

        return self


# For testing
solution_string = '''
import numpy as np

def likelihood_inference_full(
    data: dict, color_mapping: dict, indicator_mapping: dict, category_mapping: dict
) -> np.ndarray:
    """
    Generates likelihood matrix based on data observations.
    Mapping keys can be used to extract values in data dict.

    Args:
        data: A dictionary containing the observed data for each asset, with the following format:
            {
                <asset_1>: {
                    "category": <cat>,
                    "sampled_colors": {
                        <ind_1>: <color>,
                        <ind_2>: <color>,
                        ...
                },
                <asset_2>: {...},
                ...
            }
            The sampled indicator color strings correspond to the assets performance in that indicator dimension.
            The allowed string values for category, indicator, and color can be retrieved from the
            mapping arguments provided.
        color_mapping: Mapping of color names to likelihood array index.
        indicator_mapping: Mapping of indicator names to likelihood array index.
        category_mapping: Mapping of category names to likelihood array index.
    Returns:
        np.ndarray: The likelihood matrix with dimensions (n_category, n_indicator, n_color)
    """
    n_categories = len(category_mapping)
    n_indicators = len(indicator_mapping)
    n_colors = len(color_mapping)
    
    # Initialize the likelihood matrix with zeros
    likelihood_matrix = np.zeros((n_categories, n_indicators, n_colors))

    # Count occurrences of each (category, indicator, color) triplet
    count_matrix = np.zeros((n_categories, n_indicators, n_colors))

    # Iterate over each asset in the data
    for asset, asset_data in data.items():
        category = asset_data['category']
        sampled_colors = asset_data['sampled_colors']
        
        # Map the category to its index
        category_idx = category_mapping[category]
        
        # Iterate over each indicator and its observed color
        for indicator, color in sampled_colors.items():
            # Map the indicator and color to their indices
            indicator_idx = indicator_mapping[indicator]
            color_idx = color_mapping[color]
            
            # Update the count matrix
            count_matrix[category_idx, indicator_idx, color_idx] += 1

    # Normalize the counts to get likelihoods
    for category_idx in range(n_categories):
        for indicator_idx in range(n_indicators):
            total = np.sum(count_matrix[category_idx, indicator_idx, :])
            if total > 0:
                likelihood_matrix[category_idx, indicator_idx, :] = count_matrix[category_idx, indicator_idx, :] / total

    return likelihood_matrix

'''

TASK_HARD = """
We are trying to categorize a company into one of K classes and we observe J categorical indicators.  
Each indicator can take one of C colors, that characterize the company's value in the dimension of that indicator.
Each indicator has a different set of likelihoods given classes. We represent the sampling distributions as

k ~ Categorical(p)
x_j ~ Categorical(A_{j,:,k}) # The array A (J x C x K) contains the likelihoods of each indicator given each of the latent company class k.

The goal is to update an indicator likelihood prior using Bayesian inference, from N observations of indicator colors and 
categories. Each indicator has a different set of likelihoods given classes.

Here is a description of the input data, and the requirement of the output format, which will also be provided to
the code generator model:

    Args:
        data: A dictionary containing the observed data for each asset, with the following format:
            {
                <asset_1>: {
                    "category": <cat>,
                    "sampled_colors": {
                        <ind_1>: <color>,
                        <ind_2>: <color>,
                        ...
                },
                <asset_2>: {...},
                ...
            }
            The sampled indicator color strings correspond to the assets performance in that indicator dimension.
            The allowed string values for category, indicator, and color can be retrieved from the
            mapping arguments provided.
        color_mapping: Mapping of color names to likelihood array index.
        indicator_mapping: Mapping of indicator names to likelihood array index.
        category_mapping: Mapping of category names to likelihood array index.

Returns:
    np.ndarray: The likelihood matrix with dimensions (n_category, n_indicator, n_color)

"""

TASK_EASY = TASK_HARD + """

Tip:
    First work out the closed-form Bayesian update for the class probability vector p.
    Then extend this to assume the likelihood matrix is also unknown and we must simultaneously infer its parameters.
    
Recall that the unnormalized Bayesian posterior is:
posterior = likelihood * prior
where the likelihood is the indicator likelihood given the class and the prior is the prior distribution of classes.

"""


##############################
# For reference and comparison

def likelihood_inference_soln(
    data: dict, color_mapping: dict, indicator_mapping: dict, category_mapping: dict
) -> np.ndarray:
    """
    Generates likelihood matrix based on data observations.
    Mapping keys can be used to extract values in data dict.

    Args:
        data: A dictionary containing the observed data for each asset, with the following format:
            {
                <asset_1>: {
                    "category": <cat>,
                    "sampled_colors": {
                        <ind_1>: <color>,
                        <ind_2>: <color>,
                        ...
                },
                <asset_2>: {...},
                ...
            }
            The sampled indicator color strings correspond to the assets performance in that indicator dimension.
            The allowed string values for category, indicator, and color can be retrieved from the
            mapping arguments provided.
        color_mapping: Mapping of color names to likelihood array index.
        indicator_mapping: Mapping of indicator names to likelihood array index.
        category_mapping: Mapping of category names to likelihood array index.
    Returns:
        np.ndarray: The likelihood matrix with dimensions (n_category, n_indicator, n_color)
    """
    n_categories = len(category_mapping)
    n_indicators = len(indicator_mapping)
    n_colors = len(color_mapping)

    # Initialize the likelihood matrix with zeros
    likelihood_matrix = np.zeros((n_categories, n_indicators, n_colors))

    # Count occurrences of each (category, indicator, color) triplet
    # NOTE: It's possible this should be np.ones.
    count_matrix = np.zeros((n_categories, n_indicators, n_colors))

    # Iterate over each asset in the data
    for asset, asset_data in data.items():
        category = asset_data['category']
        sampled_colors = asset_data['sampled_colors']

        # Map the category to its index
        category_idx = category_mapping[category]

        # Iterate over each indicator and its observed color
        for indicator, color in sampled_colors.items():
            # Map the indicator and color to their indices
            indicator_idx = indicator_mapping[indicator]
            color_idx = color_mapping[color]

            # Update the count matrix
            count_matrix[category_idx, indicator_idx, color_idx] += 1

    # Normalize the counts to get likelihoods
    for category_idx in range(n_categories):
        for indicator_idx in range(n_indicators):
            total = np.sum(count_matrix[category_idx, indicator_idx, :])
            if total > 0:
                likelihood_matrix[category_idx, indicator_idx, :] = count_matrix[category_idx, indicator_idx, :] / total

    return likelihood_matrix


def indicator_likelihood_old1(data: dict) -> np.ndarray:
    """
    Updates the likelihood matrix based on the observed data.

    Args:
        data: A dictionary containing the observed data for each asset. The keys are the asset names,
            and the values include dictionaries with 'category' and 'sampled_colors' keys. Each 'category' item contains
            the category label of the asset, and the 'sampled_colors' item is a dictionary keyed by indicator name,
            and observed color (str). Colors are one of three ['red', 'yellow', 'green'], corresponding to
            the asset's performance in this indicator dimension.
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


def indicator_likelihood_alt(data: dict) -> np.ndarray:
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
    X = np.zeros((N, J), dtype=int)

    for i, (asset_name, asset_data) in enumerate(data.items()):
        for j, indicator in enumerate(indicators):
            x_ij = asset_data['sampled_colors'][indicator]
            x_ij = 0 if x_ij == 'red' else 1 if x_ij == 'yellow' else 2  # Convert color to integer
            X[i, j] = x_ij

    # Update beta (likelihood matrix).
    prior = np.ones((n_category, J, n_color))
    posterior = prior.copy()
    for j in range(J):  # Iterate over indicators
        for k_val, category in enumerate(categories):  # Iterate over categories
            for c in range(prior.shape[2]):  # Iterate over colors
                posterior[k_val, j, c] = prior[k_val, j, c] + np.sum(
                    (X[:, j] == c) & (np.array([asset['category'] for asset in data.values()]) == category)
                )
    epsilon = 1e-10  # Small value to prevent division by zero
    normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
    return normalized
