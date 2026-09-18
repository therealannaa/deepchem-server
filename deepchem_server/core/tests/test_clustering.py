import pandas as pd
from deepchem_server.core import config
from deepchem_server.core.common.cards import DataCard
from deepchem_server.core.primitives.clustering import cluster


def test_cluster_basic(disk_datastore):
    """Test basic clustering primitive functionality."""
    df = pd.DataFrame({'smiles': ["CCC", "CCCCC", "CCO", "CCN", "CC(=O)O", "c1ccccc1"]})
    config.set_datastore(disk_datastore)
    card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    data_address = disk_datastore.upload_data_from_memory(df, "test.csv", card)

    prediction_address, cluster_center_address = cluster(dataset_address=data_address,
                                                         num_clusters=2,
                                                         output="cluster_test",
                                                         column="smiles")

    # Get outputs from datastore
    pred_df = disk_datastore.get(prediction_address)
    center_df = disk_datastore.get(cluster_center_address)

    # Check outputs conform to expected schema
    assert pred_df is not None
    assert "cluster" in pred_df.columns
    assert "smiles_molecule" in pred_df.columns
    assert "smiles_cluster_center" in pred_df.columns

    assert center_df is not None
    assert center_df.shape[0] == 2
    assert "cluster" in center_df.columns
    assert "smiles" in center_df.columns
