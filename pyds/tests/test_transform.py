"""
Unit tests for Transform primitive using live server.
"""

import time
import uuid
import pytest

from pyds.data import Data
from pyds.primitives import Transform
from pyds.settings import Settings


class TestTransform:
    """Unit tests for Transform primitive."""

    def test_init(self, test_settings: Settings) -> None:
        """Test Transform initialization."""
        client = Transform(settings=test_settings)
        assert client.settings == test_settings
        assert client.base_url == "http://localhost:8000"

    def test_run_success(
        self,
        live_transform_client: Transform,
        live_data_client: Data,
        simple_regression_csv: str,
    ) -> None:
        """Test successful transform run on live server."""
        # Generate unique identifiers to avoid naming conflicts
        test_id = str(uuid.uuid4())[:8]
        timestamp = str(int(time.time()))

        # Upload initial test data
        upload_result = live_data_client.upload_data(
            file_path=simple_regression_csv,
            filename=f"test_transform_{test_id}_{timestamp}.csv",
            description="Test data for transform",
        )
        dataset_address = upload_result["dataset_address"]

        # Test log transformation
        result = live_transform_client.run(dataset_address=dataset_address,
                                           transform_type="log",
                                           column_name="property",
                                           new_column_name="property_transformed",
                                           output_key=f"test_transform_output_{test_id}_{timestamp}")

        assert "transformed_file_address" in result

    def test_run_missing_settings(self, test_settings_not_configured: Settings) -> None:
        """Test transform run with missing settings."""
        client = Transform(settings=test_settings_not_configured)

        with pytest.raises(ValueError, match="Missing required settings"):
            client.run(
                dataset_address="test/dataset",
                transform_type="log",
                column_name="property",
                new_column_name="property_log",
                output_key="transform_output",
            )
