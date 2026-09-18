"""
HyperparamOpt primitive module for DeepChem Server.

Contains the HyperparamOpt class for performing hyperparameter searches.
"""

from typing import Any, Dict, Optional

from .base import Primitive


class HyperparamOpt(Primitive):
    """
    Primitive for hyperparameter optimization tasks.

    This class handles submitting hyperparameter optimization searches to the DeepChem Server API.
    """

    def run(
        self,
        model_type: str,
        train_address: str,
        valid_address: str,
        hyperparams: Dict[str, Any],
        output_prefix: str,
        algorithm: str = "grid",
        metric: str = "pearson_r2_score",
        nb_epoch: int = 10,
        profile_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the hyperparameter optimization primitive.

        Args:
            model_type: Model type name recognizable by deepchem_server
            train_address: Datastore address of the training dataset
            valid_address: Datastore address of the validation dataset
            hyperparams: Dict of hyperparameter ranges
            output_prefix: Output prefix for best model and search results
            algorithm: Optimization algorithm (Default: grid)
            metric: Evaluation metric to optimize
            nb_epoch: Epochs to train each candidate
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)

        Returns:
            Response containing model_address, best_hyperparams_address, all_results_address

        Raises:
            ValueError: If required settings are missing
            requests.exceptions.RequestException: If API request fails
        """

        profile, project = self.validate_common_params(profile_name, project_name)

        data = {
            "profile_name": profile,
            "project_name": project,
            "model_type": model_type,
            "train_address": train_address,
            "valid_address": valid_address,
            "hyperparams": hyperparams,
            "output_prefix": output_prefix,
            "algorithm": algorithm,
            "metric": metric,
            "nb_epoch": nb_epoch,
        }

        response = self._post("/primitive/hyperparam-opt", json=data, headers={"Content-Type": "application/json"})
        return self._validate_response(response)
