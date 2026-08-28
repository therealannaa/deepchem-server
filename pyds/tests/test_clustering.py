"""
Unit tests for Clustering primitive using live server.
"""

import time
import uuid
import pytest

from pyds.data import Data
from pyds.primitives import Clustering
from pyds.settings import Settings


class TestClustering:
    """Unit tests for Clustering primitive."""

    def test_init(self, test_settings: Settings) -> None:
        """Test Clustering initialization."""
        client = Clustering(settings=test_settings)
        assert client.settings == test_settings
        assert client.base_url == "http://localhost:8000"

    def test_run_success(
        self,
        live_clustering_client: Clustering,
        live_data_client: Data,
        small_classification_csv: str,
    ) -> None:
        """Test successful cluster run on live server."""
        # Generate unique identifiers to avoid naming conflicts
        test_id = str(uuid.uuid4())[:8]
        timestamp = str(int(time.time()))

        # Upload initial test data
        upload_result = live_data_client.upload_data(
            file_path=small_classification_csv,
            filename=f"test_cluster_{test_id}_{timestamp}.csv",
            description="Test data for clustering",
        )
        dataset_address = upload_result["dataset_address"]

        # Run clustering
        result = live_clustering_client.run(dataset_address=dataset_address,
                                            num_clusters=2,
                                            column="smiles",
                                            output=f"test_cluster_output_{test_id}_{timestamp}")

        assert "prediction_address" in result
        assert "cluster_center_address" in result

    def test_run_missing_settings(self, test_settings_not_configured: Settings) -> None:
        """Test cluster run with missing settings."""
        client = Clustering(settings=test_settings_not_configured)

        with pytest.raises(ValueError, match="Missing required settings"):
            client.run(
                dataset_address="test/dataset",
                num_clusters=2,
                column="smiles",
                output="cluster_output",
            )
