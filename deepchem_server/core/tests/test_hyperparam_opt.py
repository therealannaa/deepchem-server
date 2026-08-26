import pandas as pd
import deepchem as dc
from deepchem_server.core import config
from deepchem_server.core.common.cards import DataCard
from deepchem_server.core.primitives.feat import featurize
from deepchem_server.core.primitives.hyperparam_opt import hyperparam_opt


def test_hyperparam_opt_basic(disk_datastore):
    """Test basic model training functionality."""
    config.set_datastore(disk_datastore)
    df = pd.DataFrame([["CCC", 0.1], ["CCCCC", 0.5]], columns=["smiles", "label"])
    card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    data_address = disk_datastore.upload_data_from_memory(df, "test.csv", card)

    # Featurize dataset as it needs to be a DeepChem dataset
    dataset_address = featurize(data_address,
                                featurizer="ecfp",
                                output="feat_test",
                                dataset_column="smiles",
                                label_column="label")

    model_address, best_hp_address, all_res_address = hyperparam_opt(model_type='linear_regression',
                                                                     train_address=dataset_address,
                                                                     valid_address=dataset_address,
                                                                     hyperparams={"fit_intercept": [True, False]},
                                                                     output_prefix="hopt_test",
                                                                     metric="pearson_r2_score",
                                                                     nb_epoch=1)

    # Loaded back model evaluation
    model = disk_datastore.get_model(model_address)
    assert isinstance(model, dc.models.Model)

    # Loaded back best params evaluation
    best_hp = disk_datastore.get(best_hp_address)
    assert best_hp is not None
    assert "fit_intercept" in best_hp

    # Loaded back all results evaluation
    all_res = disk_datastore.get(all_res_address)
    assert all_res is not None
    assert len(all_res) > 0
