import requests
from typing import Dict, Any, Optional

from pyds.primitives.base import Primitive
from pyds.settings import settings


class EvoHyperparamOpt(Primitive):
    """
    Primitive for Evolutionary Hyperparameter Optimization.
    """
    def __init__(self):
        super().__init__()

    def run(self,
            model_type: str,
            train_address: str,
            valid_address: str,
            hyperparams_space: Dict,
            output_prefix: str,
            metric: str = "pearson_r2_score",
            population_size: int = 4,
            generations: int = 3,
            nb_epoch: int = 10,
            max_evals: int = 100,
            seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs evolutionary hyperparameter search.
        """
        url = f"{settings.DEEPCHEM_SERVER_URL}:{settings.DEEPCHEM_SERVER_PORT}/primitive/evo-hyperparam-opt"
        
        payload = {
            "profile_name": settings.PROFILE_NAME,
            "project_name": settings.PROJECT_NAME,
            "model_type": model_type,
            "train_address": train_address,
            "valid_address": valid_address,
            "hyperparams_space": hyperparams_space,
            "output_prefix": output_prefix,
            "metric": metric,
            "population_size": population_size,
            "generations": generations,
            "nb_epoch": nb_epoch,
            "max_evals": max_evals,
        }
        
        if seed is not None:
            payload["seed"] = seed

        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"EvoHyperparamOpt Failed with status code: {response.status_code}, error: {response.text}")
            
        return response.json()
