import os
import tempfile
import numpy as np
import pandas as pd
import deepchem as dc
from deepchem_server.core.common import config
from deepchem_server.core.common.cards import DataCard
from deepchem_server.core.common.address import DeepchemAddress

import logging


logger = logging.getLogger(__name__)


def cluster(dataset_address: str, num_clusters: int, output: str, column: str):
    """Clusters the provided dataset and write results to datastore.

    Parameters
    ----------
    dataset_address: str
      The address of dataset to cluster.
    num_clusters: int
      Number of clusters
    output: str
      Output file prefix containing cluster center prediction and clustering results.
    column: str
      The name of SMILES column to cluster
    """
    num_clusters = int(num_clusters)
    # Steps:
    #     1. Collect dataset
    #     3. Featurize via circular fingerprint the dataset
    #     4. Perform clustering
    #     5. Upload dataset to datastore
    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError("Datastore not set")
    datacard = datastore.get(dataset_address + '.cdc', kind='data')
    assert datacard.data_type == 'pandas.DataFrame', 'clustering is supported only for pandas dataframe'
    tempdir = tempfile.TemporaryDirectory()
    temp_filename = os.path.join(tempdir.name, 'temp.csv')
    datastore.download_object(dataset_address, temp_filename)
    df = pd.read_csv(temp_filename)

    # Clustering
    featurizer = dc.feat.CircularFingerprint()
    features = featurizer.featurize(df[column])

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=num_clusters)
    cluster = kmeans.fit(features)

    df_pred = df[[column]].copy()
    df_pred['cluster'] = cluster.predict(features)

    from scipy.spatial.distance import cdist
    distances = cdist(cluster.cluster_centers_, features)
    argmin = np.argmin(distances, axis=1)
    df_centers = df.iloc[argmin][[column]].copy()
    df_centers['cluster'] = np.arange(df_centers.shape[0])

    df_pred = pd.merge(df_centers, df_pred, on='cluster', how='inner', suffixes=['_cluster_center', '_molecule'])

    card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    prediction_address = datastore.upload_data_from_memory(df_pred,
                                                           DeepchemAddress.get_key(output) + '_cluster_prediction.csv',
                                                           card)
    card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    cluster_center_address = datastore.upload_data_from_memory(df_centers,
                                                               DeepchemAddress.get_key(output) + '_cluster_centers.csv',
                                                               card)
    return (prediction_address, cluster_center_address)
