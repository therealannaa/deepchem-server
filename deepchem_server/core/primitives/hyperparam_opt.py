"""Hyperparameter search"""
import ast
import json
from typing import Dict, Optional
import deepchem as dc
from deepchem_server.core.common import config, model_mappings
from deepchem_server.core.common.cards import ModelCard, DataCard
from deepchem_server.core.common.address import DeepchemAddress
from deepchem_server.core.primitives.evaluator import deepchem_server_metrics


MIN_METRIC_LIST = ["rms_score", "mae_error"]


def hyperparam_opt(model_type: str,
                   train_address: str,
                   valid_address: str,
                   hyperparams: Dict,
                   output_prefix: str,
                   algorithm: Optional[str] = 'grid',
                   metric: str = 'pearson_r2_score',
                   nb_epoch: int = 10):
    # TODO use_max or use_min in scores
    # We should make hyperparams auto-generation default config
    if isinstance(hyperparams, str):
        hyperparams = ast.literal_eval(hyperparams)
    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError("Datastore not set")
    train_dataset = datastore.get(train_address)
    valid_dataset = datastore.get(valid_address)

    if model_type not in model_mappings.model_address_map:
        raise ValueError("Model type not recognized.")

    def _model_builder(**model_params):
        model = model_mappings.model_address_map[model_type](**model_params)
        return model

    optimizer = dc.hyper.GridHyperparamOpt(_model_builder)

    if metric in MIN_METRIC_LIST:
        use_max: bool = False
    else:
        use_max = True

    metric_obj = deepchem_server_metrics[metric]
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(hyperparams,
                                                                            train_dataset,
                                                                            valid_dataset,
                                                                            metric_obj,
                                                                            use_max=use_max,
                                                                            nb_epoch=nb_epoch)

    model_card = ModelCard(address='',
                           model_type=model_type,
                           train_dataset_address=train_address,
                           valid_dataset_address=valid_address,
                           init_kwargs=best_hyperparams,
                           train_kwargs={})
    model_name = DeepchemAddress.get_key(output_prefix) + '_best_model'
    model_address = datastore.upload_data_from_memory(best_model, model_name, model_card, kind='model')

    best_hyperparams = json.dumps(best_hyperparams)
    description = f"best hyperparams from {model_type} model on {train_address} train dataset and {valid_address} valid dataset"
    card = DataCard(address='', file_type='json', data_type='json', description=description)
    if datastore is None:
        raise ValueError("Datastore not set")
    output_address_best_hyperparams = datastore.upload_data_from_memory(
        best_hyperparams,
        DeepchemAddress.get_key(output_prefix) + '_best_hyperparams.json', card)

    all_results = json.dumps(all_results)
    description = f"all results of hyperparams search on {model_type} model using {train_address} train dataset and {valid_address} valid dataset"
    card = DataCard(address='', file_type='json', data_type='json', description=description)
    output_address = datastore.upload_data_from_memory(all_results,
                                                       DeepchemAddress.get_key(output_prefix) + '_all_results.json',
                                                       card)
    return model_address, output_address_best_hyperparams, output_address
