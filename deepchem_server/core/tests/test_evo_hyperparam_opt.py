import pandas as pd
import deepchem as dc
from deepchem_server.core import config
from deepchem_server.core.common.cards import DataCard
from deepchem_server.core.primitives.feat import featurize
from deepchem_server.core.primitives.hyperparam_opt import hyperparam_opt
from deepchem_server.core.primitives.evo_hyperparam_opt import evo_hyperparam_opt
import json


def test_evo_hyperparam_opt_vs_grid(disk_datastore):
    """
    Test and benchmark evolutionary hyperparameter optimization
    against grid search under a small budget constraint.
    """
    config.set_datastore(disk_datastore)
    
    # Toy regression dataset
    df = pd.DataFrame([["C", 0.1], ["CC", 0.5], ["CCC", 0.9], ["CCCC", 1.3], ["CCCCC", 1.8], ["CCCCCC", 2.2]], 
                      columns=["smiles", "label"])
    card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    data_address = disk_datastore.upload_data_from_memory(df, "test.csv", card)

    # Featurize dataset
    dataset_address = featurize(data_address,
                                featurizer="ecfp",
                                output="feat_test",
                                dataset_column="smiles",
                                label_column="label")

    # Defined large search space
    search_space = {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5],
    }
    # Total combinations = 3 * 3 * 2 = 18 configurations.

    # 1. Run Grid Search
    # Note: GridSearch runs exhaustively unless externally bounded; 
    # the server's basic hyperparam_opt just runs the whole grid.
    grid_model_addr, grid_best_hp_addr, grid_all_res_addr = hyperparam_opt(
        model_type='random_forest_regressor',
        train_address=dataset_address,
        valid_address=dataset_address,
        hyperparams=search_space,
        output_prefix="grid_test",
        metric="pearson_r2_score",
        nb_epoch=1
    )
    
    # Get best grid score
    grid_best_hp = disk_datastore.get(grid_best_hp_addr)
    # The current standard deepchem GridHyperparamOpt unit tests 
    # do not output the score via the primitive, but it logs all results.
    grid_all_results = disk_datastore.get(grid_all_res_addr)
    # Sort Grid scores
    sorted_grid_scores = sorted(grid_all_results.values(), reverse=True)
    grid_best_score = sorted_grid_scores[0] if sorted_grid_scores else float('-inf')


    # 2. Run Evo Search constrained to a small budget (max_evals)
    # We constrain max_evals to something much smaller than 18 (e.g. 8)
    evo_model_addr, evo_best_hp_addr, evo_all_res_addr = evo_hyperparam_opt(
        model_type='random_forest_regressor',
        train_address=dataset_address,
        valid_address=dataset_address,
        hyperparams_space=search_space,
        output_prefix="evo_test",
        metric="pearson_r2_score",
        population_size=4,
        generations=3,
        nb_epoch=1,
        max_evals=8,
        seed=42
    )

    # Loaded back model evaluation
    evo_model = disk_datastore.get_model(evo_model_addr)
    assert isinstance(evo_model, dc.models.Model)

    # Check best params format and score
    evo_best_hp_data = disk_datastore.get(evo_best_hp_addr)
    assert 'params' in evo_best_hp_data
    assert 'score' in evo_best_hp_data
    assert 'config_id' in evo_best_hp_data

    # Ensure total evaluations didn't exceed budget
    evo_all_results = disk_datastore.get(evo_all_res_addr)
    assert len(evo_all_results) <= 8
    
    # Check that lineage (parent_id) is being tracked properly
    assert all('parent_id' in res for res in evo_all_results)
