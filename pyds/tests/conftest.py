"""
Pytest configuration and fixtures for pyds tests.
"""

import os
from typing import Any, Dict, Generator, List

import pytest
import responses

from pyds import Settings
from pyds.base.client import BaseClient
from pyds.data import Data
from pyds.primitives import DelDenoise, Evaluate, Featurize, Infer, Partition, Train, HyperparamOpt

from .test_utils import (
    cleanup_temp_file,
    create_classification_csv,
    create_regression_csv,
    create_test_settings_file,
    get_complex_regression_dataset,
    get_large_classification_dataset,
    get_minimal_classification_dataset,
    get_simple_regression_dataset,
    get_small_classification_dataset,
)


@pytest.fixture
def temp_settings_file() -> Generator[str, None, None]:
    """Create a temporary settings file for testing."""
    settings_file = create_test_settings_file()

    yield settings_file

    # Clean up
    cleanup_temp_file(settings_file)


@pytest.fixture
def test_settings(temp_settings_file: str) -> Settings:
    """Create test settings instance."""
    settings = Settings(
        settings_file=temp_settings_file,
        profile="test_profile",
        project="test_project",
        base_url="http://localhost:8000",
    )
    return settings


@pytest.fixture
def test_settings_not_configured(temp_settings_file: str) -> Settings:
    """Create test settings instance without profile/project configured."""
    settings = Settings(settings_file=temp_settings_file, base_url="http://localhost:8000")
    return settings


@pytest.fixture
def base_client(test_settings: Settings) -> BaseClient:
    """Create BaseClient instance for testing."""
    return BaseClient(settings=test_settings)


@pytest.fixture
def featurize_client(test_settings: Settings) -> Featurize:
    """Create Featurize primitive client for testing."""
    return Featurize(settings=test_settings)


@pytest.fixture
def train_client(test_settings: Settings) -> Train:
    """Create Train primitive client for testing."""
    return Train(settings=test_settings)


@pytest.fixture
def evaluate_client(test_settings: Settings) -> Evaluate:
    """Create Evaluate primitive client for testing."""
    return Evaluate(settings=test_settings)


@pytest.fixture
def infer_client(test_settings: Settings) -> Infer:
    """Create Infer primitive client for testing."""
    return Infer(settings=test_settings)


@pytest.fixture
def partition_client(test_settings: Settings) -> Partition:
    """Create Partition primitive client for testing."""
    return Partition(settings=test_settings)


@pytest.fixture
def del_denoise_client(test_settings: Settings) -> DelDenoise:
    """Create DelDenoise primitive client for testing."""
    return DelDenoise(settings=test_settings)


@pytest.fixture
def hyperparam_opt_client(test_settings: Settings) -> HyperparamOpt:
    """Create HyperparamOpt primitive client for testing."""
    return HyperparamOpt(settings=test_settings)


@pytest.fixture
def data_client(test_settings: Settings) -> Data:
    """Create Data client for testing."""
    return Data(settings=test_settings)


@pytest.fixture
def temp_test_file() -> Generator[str, None, None]:
    """Create a temporary test file for upload testing."""
    # Use the same data as small_classification_csv for consistency
    test_file = create_classification_csv(size="small")

    yield test_file

    # Clean up
    cleanup_temp_file(test_file)


@pytest.fixture
def mock_responses() -> Generator[responses.RequestsMock, None, None]:
    """Setup responses mock for HTTP requests."""
    with responses.RequestsMock() as rsps:
        yield rsps


@pytest.fixture
def sample_featurize_response() -> Dict[str, Any]:
    """Sample response for featurize endpoint."""
    return {"featurized_file_address": "test_profile/test_project/datasets/featurized_data"}


@pytest.fixture
def sample_train_response() -> Dict[str, Any]:
    """Sample response for train endpoint."""
    return {"trained_model_address": "test_profile/test_project/models/trained_model"}


@pytest.fixture
def sample_evaluate_response() -> Dict[str, Any]:
    """Sample response for evaluate endpoint."""
    return {"evaluation_result_address": "test_profile/test_project/evaluations/eval_result"}


@pytest.fixture
def sample_infer_response() -> Dict[str, Any]:
    """Sample response for infer endpoint."""
    return {"inference_results_address": "test_profile/test_project/predictions/infer_result"}


@pytest.fixture
def sample_upload_response() -> Dict[str, Any]:
    """Sample response for upload endpoint."""
    return {"dataset_address": "test_profile/test_project/datasets/uploaded_data"}


@pytest.fixture
def sample_healthcheck_response() -> Dict[str, Any]:
    """Sample response for healthcheck endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# Common test data fixtures
@pytest.fixture
def test_dataset_address() -> str:
    """Test dataset address."""
    return "test_profile/test_project/datasets/test_data"


@pytest.fixture
def test_model_address() -> str:
    """Test model address."""
    return "test_profile/test_project/models/test_model"


@pytest.fixture
def test_featurizer() -> str:
    """Test featurizer name."""
    return "CircularFingerprint"


@pytest.fixture
def test_model_type() -> str:
    """Test model type."""
    return "MultitaskClassifier"


@pytest.fixture
def test_metrics() -> List[str]:
    """Test metrics list."""
    return ["roc_auc_score", "accuracy_score"]


# Live server testing fixtures
@pytest.fixture
def live_server_url() -> str:
    """URL for live server testing."""
    return os.getenv("PYDS_TEST_SERVER_URL", "http://localhost:8000")


@pytest.fixture
def live_settings(temp_settings_file: str, live_server_url: str) -> Settings:
    """Create settings for live server testing."""
    return Settings(
        settings_file=temp_settings_file,
        profile=os.getenv("PYDS_TEST_PROFILE", "test_profile"),
        project=os.getenv("PYDS_TEST_PROJECT", "test_project"),
        base_url=live_server_url,
    )


@pytest.fixture
def live_base_client(live_settings: Settings) -> BaseClient:
    """Create BaseClient for live server testing."""
    return BaseClient(settings=live_settings)


@pytest.fixture
def live_featurize_client(live_settings: Settings) -> Featurize:
    """Create Featurize client for live server testing."""
    return Featurize(settings=live_settings)


@pytest.fixture
def live_train_client(live_settings: Settings) -> Train:
    """Create Train client for live server testing."""
    return Train(settings=live_settings)


@pytest.fixture
def live_evaluate_client(live_settings: Settings) -> Evaluate:
    """Create Evaluate client for live server testing."""
    return Evaluate(settings=live_settings)


@pytest.fixture
def live_infer_client(live_settings: Settings) -> Infer:
    """Create Infer client for live server testing."""
    return Infer(settings=live_settings)


@pytest.fixture
def live_partition_client(live_settings: Settings) -> Partition:
    """Create Partition client for live server testing."""
    return Partition(settings=live_settings)


@pytest.fixture
def live_del_denoise_client(live_settings: Settings) -> DelDenoise:
    """Create DelDenoise client for live server testing."""
    return DelDenoise(settings=live_settings)


@pytest.fixture
def live_hyperparam_opt_client(live_settings: Settings) -> HyperparamOpt:
    """Create HyperparamOpt client for live server testing."""
    return HyperparamOpt(settings=live_settings)


# ===========================
# Common Dataset Fixtures
# ===========================


@pytest.fixture
def small_classification_csv() -> Generator[str, None, None]:
    """Create a small classification dataset CSV file (3 molecules)."""
    csv_file = create_classification_csv(size="small")
    yield csv_file
    cleanup_temp_file(csv_file)


@pytest.fixture
def minimal_classification_csv() -> Generator[str, None, None]:
    """Create a minimal classification dataset CSV file (5 molecules)."""
    csv_file = create_classification_csv(size="minimal")
    yield csv_file
    cleanup_temp_file(csv_file)


@pytest.fixture
def large_classification_csv() -> Generator[str, None, None]:
    """Create a large classification dataset CSV file (15 molecules)."""
    csv_file = create_classification_csv(size="large")
    yield csv_file
    cleanup_temp_file(csv_file)


@pytest.fixture
def simple_regression_csv() -> Generator[str, None, None]:
    """Create a simple regression dataset CSV file."""
    csv_file = create_regression_csv(complexity="simple", headers=["smiles", "property"])
    yield csv_file
    cleanup_temp_file(csv_file)


@pytest.fixture
def complex_regression_csv() -> Generator[str, None, None]:
    """Create a complex regression dataset CSV file."""
    csv_file = create_regression_csv(complexity="complex", headers=["smiles", "target"])
    yield csv_file
    cleanup_temp_file(csv_file)


@pytest.fixture
def small_classification_data() -> list[tuple[str, int]]:
    """Get small classification dataset as raw data."""
    return get_small_classification_dataset()


@pytest.fixture
def minimal_classification_data() -> list[tuple[str, int]]:
    """Get minimal classification dataset as raw data."""
    return get_minimal_classification_dataset()


@pytest.fixture
def large_classification_data() -> list[tuple[str, int]]:
    """Get large classification dataset as raw data."""
    return get_large_classification_dataset()


@pytest.fixture
def simple_regression_data() -> list[tuple[str, float]]:
    """Get simple regression dataset as raw data."""
    return get_simple_regression_dataset()


@pytest.fixture
def complex_regression_data() -> list[tuple[str, float]]:
    """Get complex regression dataset as raw data."""
    return get_complex_regression_dataset()


@pytest.fixture
def live_data_client(live_settings: Settings) -> Data:
    """Create Data client for live server testing."""
    return Data(settings=live_settings)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    pass
