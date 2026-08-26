"""
Unit tests for HyperparamOpt primitive using live server.
"""

import time
from typing import Any
import uuid
import pytest

from pyds.data import Data
from pyds.primitives import HyperparamOpt
from pyds.settings import Settings


class TestHyperparamOpt:
    """Unit tests for HyperparamOpt primitive."""

    def test_init(self, test_settings: Settings) -> None:
        """Test HyperparamOpt initialization."""
        client = HyperparamOpt(settings=test_settings)
        assert client.settings == test_settings
        assert client.base_url == "http://localhost:8000"

    def test_run_success(
        self,
        live_hyperparam_opt_client: HyperparamOpt,
        live_data_client: Data,
        live_featurize_client: Any,
        simple_regression_csv: str,
    ) -> None:
        """Test successful hyperparameter optimization run on live server."""
        # Generate unique identifiers to avoid naming conflicts
        test_id = str(uuid.uuid4())[:8]
        timestamp = str(int(time.time()))

        # Upload train mock dataset
        upload_train = live_data_client.upload_data(
            file_path=simple_regression_csv,
            filename=f"test_hopt_train_{test_id}_{timestamp}.csv",
            description="Train data for hyperparam search",
        )
        train_address = upload_train["dataset_address"]

        # Upload valid mock dataset
        upload_valid = live_data_client.upload_data(
            file_path=simple_regression_csv,
            filename=f"test_hopt_valid_{test_id}_{timestamp}.csv",
            description="Validation data for hyperparam search",
        )
        valid_address = upload_valid["dataset_address"]

        # Featurize train dataset
        feat_train = live_featurize_client.run(
            dataset_address=train_address,
            featurizer="ecfp",
            output=f"test_hopt_train_feat_{test_id}_{timestamp}",
            dataset_column="smiles",
            label_column="property",
        )
        train_feat_address = feat_train["featurized_file_address"]

        # Featurize valid dataset
        feat_valid = live_featurize_client.run(
            dataset_address=valid_address,
            featurizer="ecfp",
            output=f"test_hopt_valid_feat_{test_id}_{timestamp}",
            dataset_column="smiles",
            label_column="property",
        )
        valid_feat_address = feat_valid["featurized_file_address"]

        # Run hyperparam_opt through live api wrapper
        result = live_hyperparam_opt_client.run(
            model_type="linear_regression",
            train_address=train_feat_address,
            valid_address=valid_feat_address,
            hyperparams={"fit_intercept": [True, False]},
            output_prefix=f"test_hopt_output_{test_id}_{timestamp}",
            metric="pearson_r2_score",
            nb_epoch=1,
        )

        assert "model_address" in result
        assert "best_hyperparams_address" in result
        assert "all_results_address" in result

    def test_run_missing_settings(self, test_settings_not_configured: Settings) -> None:
        """Test hyperparam_opt run with missing settings."""
        client = HyperparamOpt(settings=test_settings_not_configured)

        with pytest.raises(ValueError, match="Missing required settings"):
            client.run(
                model_type="linear_regression",
                train_address="test/train",
                valid_address="test/valid",
                hyperparams={"fit_intercept": [True]},
                output_prefix="hopt_output",
            )
