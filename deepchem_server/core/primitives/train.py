import ast
import logging
import math
from typing import Dict, Optional

from deepchem_server.core.common import config, model_mappings
from deepchem_server.core.common.address import DeepchemAddress
from deepchem_server.core.common.cards import ModelCard
from deepchem_server.core.common.progress_logger import log_progress


logger = logging.getLogger(__name__)


def train(model_type: str,
          dataset_address: str,
          model_name: str,
          init_kwargs: Optional[Dict] = None,
          train_kwargs: Optional[Dict] = None) -> str:
    """Trains a model on the specified dataset and writes output to datastore.

    Parameters
    ----------
    model_type: str
      A model string recognized by deepchem_server.core.model_mappings
    dataset_address: str
      The Deepchem server datastore address of the training dataset. The dataset
      in the address should be a DeepChem dataset.
    model_name: str
      The name under which the output trained model will be
      stored in the workspace.
    init_kwargs: Optional[Dict]
      Keyword arguments to pass to model on initialization.
    train_kwargs: Optional[Dict]
      Keyword arguments to pass to model on training.

    Examples
    --------
    >>> from deepchem_server.core.cards import DataCard
    >>> from deepchem_server.core import config, featurize
    >>> from deepchem_server.core.datastore import DiskDataStore
    >>> import tempfile
    >>> import pandas as pd
    >>> disk_datastore = DiskDataStore('profile', 'project', tempfile.mkdtemp())
    >>> config.set_datastore(disk_datastore)
    >>> df = pd.DataFrame([["CCC", 0], ["CCCCC", 1]], columns=["smiles", "label"])
    >>> card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    >>> data_address = disk_datastore.upload_data_from_memory(df, "test.csv", card)
    >>> dataset_address = featurize(data_address,
    ... featurizer="ecfp",
    ... output="feat_test",
    ... dataset_column="smiles",
    ... label_column="label")
    >>> train(model_type = "random_forest_regressor",
    ... dataset_address = dataset_address,
    ... model_name = "random_forest_model")
    'deepchem://profile/project/random_forest_model'

    """
    if init_kwargs is None:
        init_kwargs = {}
    if train_kwargs is None:
        train_kwargs = {}

    if isinstance(init_kwargs, str):
        init_kwargs = ast.literal_eval(init_kwargs)
    if isinstance(train_kwargs, str):
        train_kwargs = ast.literal_eval(train_kwargs)
    if model_type not in model_mappings.model_address_map:
        raise ValueError(f"Model type {model_type} not recognized")

    model = model_mappings.model_address_map[model_type](**init_kwargs)
    model_name = DeepchemAddress.get_key(model_name)
    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError("Datastore not set")

    if datastore.exists(model_name):
        raise FileExistsError(f"Model name {model_name} already exists.")

    dataset_size = datastore.get_file_size(dataset_address)
    log_progress('training', 10, f"downloading dataset '{dataset_address}' ({dataset_size} bytes)")
    dataset = datastore.get(dataset_address)

    # if the model is a TorchModel, add a callback to log the epoch number
    try:
        from deepchem.models.torch_models import TorchModel
        is_torch = isinstance(model, TorchModel)
    except ImportError:
        is_torch = False

    if is_torch:
        batch_size = init_kwargs.get('batch_size', 100)
        nb_epoch = train_kwargs.get('nb_epoch', 10)
        total_steps = math.ceil(dataset.get_shape()[0][0] / batch_size) * nb_epoch
        log_frequency = 1

        model.log_frequency = log_frequency

        def callback(_, step):
            if step % log_frequency == 0:
                log_progress('training', int(10 + (step / total_steps) * 80),
                             f"training model '{model_name}' ({step}/{total_steps})")

        model.fit(dataset, callbacks=[callback], **train_kwargs)
    else:
        # TODO Add test for deepchem dataset
        log_progress('training', 50, f"training model '{model_name}'")
        model.fit(dataset, **train_kwargs)
    try:
        # This is a no-op for non sklearn models
        model.save()
    except NotImplementedError:
        pass

    description = ''

    card = ModelCard(address='',
                     model_type=model_type,
                     train_dataset_address=dataset_address,
                     init_kwargs=init_kwargs,
                     train_kwargs=train_kwargs,
                     description=description)
    log_progress('training', 95, f"uploading model '{model_name}' to datastore")

    model_address = datastore.upload_data_from_memory(model, model_name, card, kind='model')
    log_progress('training', 100, f"model '{model_name}' uploaded to datastore")

    return model_address
