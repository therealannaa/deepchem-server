import os
import tempfile
from typing import Dict, Callable
from deepchem_server.core.common import config
from deepchem_server.core.common.cards import DataCard
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from scipy.stats import zscore
from deepchem_server.core.common.address import DeepchemAddress


def normalization_transform(df: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Normalization (standardization) transformation function

    Parameters
    ----------
    df: pd.DataFrame
        dataframe containing the colun to be transformed
    column_name: str
        name of the column to be transformed

    Returns
    -------
    pd.Series
        transformed column values

    Example
    -------
    >>> import pandas as pd
    >>> data = {'test_col': [1, 2, 3]}
    >>> df = pd.DataFrame(data)
    >>> df['new_col'] = normalization_transform(df=df, column_name='test_col')
    >>> round(df['new_col'][0], 2)
    -1.22
    """
    return zscore(df[column_name])


def log_transform(df: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Log transformation function

    Parameters
    ----------
    df: pd.DataFrame
        dataframe containing the colun to be transformed
    column_name: str
        name of the column to be transformed

    Returns
    -------
    pd.Series
        transformed column values

    Example
    -------
    >>> import pandas as pd
    >>> data = {'test_col': [1, 2, 3]}
    >>> df = pd.DataFrame(data)
    >>> df['new_col'] = log_transform(df=df, column_name='test_col')
    >>> round(df['new_col'][0], 2)
    0.69
    """
    return np.log(df[column_name] + 1)


def transform(dataset_address: str, transform_type: str, column_name: str, new_column_name: str,
              output_key: str) -> str:
    """
    `transform` primitive transforms columns based on different transform types
    and uploads the new dataset to datastore.

    The type of transform available:
    - norm
        Normalization (standardization) transformation
    - log
        Log transformation

    Parameters
    ----------
    dataset_address: str
        The Chiron address of the dataset to transform.
    transform_type: str
        The transform type (example: "log")
    column_name: str
        The name of the column to be transformed
    new_column_name: str
        The name of the column formed after transformation
    output_key: str
        The name of the new transformed dataset to stored in the datastore

    Returns
    -------
    address: str
        The Chiron address of the transformed dataset

    Example
    -------
    >>> from deepchem_server.core.common.cards import DataCard
    >>> from deepchem_server.core.common import config
    >>> from deepchem_server.core.datastore import DiskDataStore
    >>> import tempfile
    >>> import pandas as pd
    >>> disk_datastore = DiskDataStore('profile', 'project', tempfile.mkdtemp())
    >>> config.set_datastore(disk_datastore)
    >>> df = pd.DataFrame({'test_col': [1, 2, 3]})
    >>> card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    >>> data_address = disk_datastore.upload_data_from_memory(df, "test.csv", card)
    >>> plot_image_address = transform(dataset_address=data_address,
    ... transform_type='log',
    ... column_name='test_col',
    ... new_column_name='test_col_log',
    ... output_key='updated_dataset')
    """
    SUPPORTED_TRANSFORM_TYPES: Dict[str, Callable[[pd.DataFrame, str], pd.Series]] = {
        'norm': normalization_transform,
        'log': log_transform
    }

    transform_type = transform_type.lower()
    if transform_type not in SUPPORTED_TRANSFORM_TYPES:
        raise ValueError("Transform type not recognized.")

    assert dataset_address.endswith('csv')

    tempdir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
    raw_dataset_path: str = os.path.join(tempdir.name, 'temp.csv')
    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError("Datastore not set")
    datastore.download_object(dataset_address, raw_dataset_path)
    df: pd.DataFrame = pd.read_csv(raw_dataset_path)

    if column_name not in df.columns:
        raise ValueError(f"{column_name} column not in specified dataset")

    if df[column_name].isna().any():
        raise Exception(f"{column_name} column contains NaN values")

    if not is_numeric_dtype(df[column_name]):
        raise Exception(f"{column_name} column does not contain numeric data")

    df[new_column_name] = SUPPORTED_TRANSFORM_TYPES[transform_type](df=df, column_name=column_name)  # type: ignore

    output_key = DeepchemAddress.get_key(output_key)
    if not output_key.endswith('.csv'):
        output_key = output_key + '.csv'

    tempdir = tempfile.TemporaryDirectory()
    temp_output_path = os.path.join(tempdir.name, output_key)
    temp_output_dir = os.path.dirname(temp_output_path)
    if not os.path.exists(temp_output_dir):
        os.makedirs(temp_output_dir)
    df.to_csv(temp_output_path)

    card = DataCard(address='', file_type='csv', data_type='DataFrame')
    address: str = datastore.upload_data(output_key, temp_output_path, card)
    return address
