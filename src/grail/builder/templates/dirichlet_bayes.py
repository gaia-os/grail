import ast

import numpy as np
from opto.trace import bundle
from pydantic import model_validator, ValidationError

from grail.elixir.validator import ElixirValidator, GREEN_IMPORTS


class DirichletBayesFunction(ElixirValidator):
    """
    Defines a function for Bayesian updating according to a Dirichlet prior (parametrized by 'alphas').
    Input arguments include:
        1. A 1D likelihood of the observed data
        2. A 1D prior of over category beliefs
        3. Alphas that parametrize the Dirichlet prior.
    Output:
        1. Updated posterior probabilities
        2. Updated alphas
    """
    is_function = True
    function_name = "dirichlet_bayes"
    prompt_description = "Perform a Bayesian update for a Dirichlet prior, given the prior, alphas, and likelihood."
    required_args = tuple(
        [
            {
                "arg": "likelihood",
                "description": "Likelihood of the data",
                "datatype": np.ndarray
            },
            {
                "arg": "p_k",
                "description": "Class predictions",
                "datatype": np.ndarray
            },
            {
                "arg": "alphas",
                "description": "Dirichlet prior alpha parameters",
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
            },
            {
                "name": "Updated alphas",
                "description": "The alphas after updating",
                "type": np.ndarray
            },
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # "after"-model validators have 'self' argument and act on the instance object
    @model_validator(mode="after")
    def test_successful_inference(self):
        """
        Validate the it performs the inference correctly
        """
        tree = ast.parse(self.code)

        # Check for green imports and add them to the namespace
        # Also provide some standards
        namespace = {}
        # The following is to try and import green libs on the fly,
        # but it seems to be a bit buggy and the internal function fails to use them
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name.split(".")[0] in GREEN_IMPORTS:
                        # Add the import to the namespace
                        # TODO -- Why did I comment this out...
                        # namespace[alias.name.split(".")[0]] = __import__(alias.name)
                        continue

        # Create local namespace for this function
        exec(self.code, namespace)
        dirichlet_bayes = namespace[DirichletBayesFunction.function_name]
        # Import any green modules the code requires, and inject them into the function's globals
        for name, module in namespace.items():
            if name in GREEN_IMPORTS:
                dirichlet_bayes.__globals__[name] = module  # Inject into function's globals

        color_pred = np.eye(3)[0]
        # Test the function -- likelihood rows add to 1
        likelihood = np.asarray(
            [
                [0.1, 0.1, 0.8],
                [0.1, 0.6, 0.3],
                [0.5, 0.4, 0.01],
            ]
        )
        # Reduce the likelihood given the observed color
        likelihood = likelihood @ color_pred.reshape(-1, 1)
        likelihood = likelihood.reshape(3, )
        p_k = np.array([0.2, 0.2, 0.6])
        alphas = np.array([1, 1, 1])

        soln_pk = np.asarray([0.05882353, 0.05882353, 0.88235294])
        soln_alphas = np.asarray([1.05882353, 1.05882353, 1.88235294])
        eps = 1e-6

        try:
            new_pk, new_alphas = dirichlet_bayes(likelihood, p_k, alphas)
        except Exception as e:
            raise e
            # raise (
            #     "Exception occurred while testing method for test_array_shapes",
            #     exception=e,
            # )
        else:
            # Check we are within error
            if np.allclose(new_pk, soln_pk, atol=eps) and np.allclose(new_alphas, soln_alphas, atol=eps):
                # Good
                ...
            else:
                raise ValidationError(
                    "Failed test_array_shapes: returned values are outside tolerance",
                    # exception=ValidationError(),
                )

        return self


# For testing
solution_string = '''
import numpy

def dirichlet_bayes(likelihood: numpy.ndarray, p_k: numpy.ndarray, alphas: numpy.ndarray) -> tuple[numpy.ndarray, 
numpy.ndarray]:
    """
    Bayesian update of the asset's class prediction.
    Does not modify p_k and alphas in place, returns updated versions.

    :param likelihood:      Likelihood matrix for the indicator
    :param p_k:             Class predictions
    :param alphas:          Dirichlet prior alpha parameters
    :return:                Updated class prediction posterior, and alphas
    """
    ###
    # MAP method.
    # From some brief experimenting, is essentially same outcome but very slightly worse variance
    # alphas_star = alphas + posterior - 1
    # alphas += alphas_star / numpy.sum(alphas_star)
    ###
    posterior = p_k * likelihood
    posterior /= numpy.sum(posterior)
    # Avoiding modifying in place, just in case
    _alphas = alphas + posterior
    return posterior, _alphas
'''


# Add skeleton code for trace for the functions we want to generate
@bundle(trainable=True)
def dirichlet_bayes_elixir(likelihood: np.ndarray, p_k: np.ndarray, alphas: np.ndarray) -> tuple[
    np.ndarray, np.ndarray]:
    """
    Bayesian update of the asset's class prediction.
    Does not modify p_k and alphas in place, returns updated versions.

    :param likelihood:      Likelihood matrix for the indicator
    :param p_k:             Class predictions
    :param alphas:          Dirichlet prior alpha parameters
    :return:                Updated class prediction posterior, and alphas
    """

    posterior = np.ndarray()  # Placeholder for the generated code
    # Avoiding modifying in place, just in case
    _alphas = np.ndarray()  # Placeholder for the generated code
    return posterior, _alphas
