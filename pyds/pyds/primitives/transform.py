"""
Transform primitive module for DeepChem Server.

Contains the Transform class for performing dataset columns transformations.
"""

from typing import Any, Dict, Optional

from .base import Primitive


class Transform(Primitive):
    """
    Primitive for transformation tasks.

    This class handles submitting transform jobs to the DeepChem Server API.
    """

    def run(
        self,
        dataset_address: str,
        transform_type: str,
        column_name: str,
        new_column_name: str,
        output_key: str,
        profile_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the transform primitive.

        Args:
            dataset_address: Datastore address of dataset to transform
            transform_type: Transform type to use (e.g., 'log' or 'norm')
            column_name: Name of the column to transform
            new_column_name: Name of the resulting transformed column
            output_key: Name of the output transformed dataset
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)

        Returns:
            Response containing the transformed file address

        Raises:
            ValueError: If required settings are missing
            requests.exceptions.RequestException: If API request fails
        """

        profile, project = self.validate_common_params(profile_name, project_name)

        data = {
            "profile_name": profile,
            "project_name": project,
            "dataset_address": dataset_address,
            "transform_type": transform_type,
            "column_name": column_name,
            "new_column_name": new_column_name,
            "output_key": output_key,
        }

        response = self._post("/primitive/transform", json=data, headers={"Content-Type": "application/json"})
        return self._validate_response(response)
