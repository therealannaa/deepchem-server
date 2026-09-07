"""
Clustering primitive module for DeepChem Server.

Contains the Clustering class for performing dataset clustering.
"""

from typing import Any, Dict, Optional

from .base import Primitive


class Clustering(Primitive):
    """
    Primitive for clustering tasks.

    This class handles submitting clustering jobs to the DeepChem Server API.
    """

    def run(
        self,
        dataset_address: str,
        num_clusters: int,
        column: str,
        output: str,
        profile_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the clustering primitive.

        Args:
            dataset_address: Datastore address of dataset to cluster
            num_clusters: Number of clusters (k)
            column: Name of SMILES column to cluster
            output: Output file prefix
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)

        Returns:
            Response containing the prediction address and cluster centers address

        Raises:
            ValueError: If required settings are missing
            requests.exceptions.RequestException: If API request fails
        """

        profile, project = self.validate_common_params(profile_name, project_name)

        data = {
            "profile_name": profile,
            "project_name": project,
            "dataset_address": dataset_address,
            "num_clusters": num_clusters,
            "column": column,
            "output": output,
        }

        response = self._post("/primitive/cluster", json=data, headers={"Content-Type": "application/json"})
        return self._validate_response(response)
