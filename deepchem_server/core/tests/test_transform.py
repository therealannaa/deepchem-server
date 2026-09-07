import pandas as pd
import pytest
import numpy as np

from deepchem_server.core import config
from deepchem_server.core.common.cards import DataCard
from deepchem_server.core.primitives.transform import transform


def test_transform_log(disk_datastore):
    """Test log transformation functionality."""
    df = pd.DataFrame({'property': [1.0, 2.0, 3.0]})
    config.set_datastore(disk_datastore)
    card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    data_address = disk_datastore.upload_data_from_memory(df, "test.csv", card)

    dataset_address = transform(dataset_address=data_address,
                                transform_type="log",
                                column_name="property",
                                new_column_name="property_log",
                                output_key="transform_test")

    # Get from datastore
    transformed_df = disk_datastore.get(dataset_address)
    assert transformed_df is not None
    assert "property_log" in transformed_df.columns
    # log_transform returns log(x+1)
    np.testing.assert_array_almost_equal(transformed_df["property_log"].values, np.log(df["property"].values + 1))


def test_transform_norm(disk_datastore):
    """Test min-max normalization transformation functionality."""
    df = pd.DataFrame({'property': [1.0, 2.0, 3.0]})
    config.set_datastore(disk_datastore)
    card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    data_address = disk_datastore.upload_data_from_memory(df, "test.csv", card)

    dataset_address = transform(dataset_address=data_address,
                                transform_type="norm",
                                column_name="property",
                                new_column_name="property_norm",
                                output_key="transform_test_norm")

    # Get from datastore
    transformed_df = disk_datastore.get(dataset_address)
    assert transformed_df is not None
    assert "property_norm" in transformed_df.columns
    from scipy.stats import zscore
    np.testing.assert_array_almost_equal(transformed_df["property_norm"].values, zscore(df["property"].values))


def test_transform_invalid_type(disk_datastore):
    """Test invalid transform type."""
    df = pd.DataFrame({'property': [1.0, 2.0, 3.0]})
    config.set_datastore(disk_datastore)
    card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    data_address = disk_datastore.upload_data_from_memory(df, "test.csv", card)

    with pytest.raises(ValueError, match="Transform type not recognized"):
        transform(dataset_address=data_address,
                  transform_type="invalid_type",
                  column_name="property",
                  new_column_name="property_invalid",
                  output_key="transform_test_err")
