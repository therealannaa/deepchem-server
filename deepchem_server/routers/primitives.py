import ast
import json
import math
from typing import Annotated, Dict, List, Optional, Union, Any

from fastapi import APIRouter, HTTPException
from fastapi.params import Body

from deepchem_server.core.common import model_mappings
from deepchem_server.core.primitives.feat import featurizer_map
from deepchem_server.utils import parse_boolean_none_values_from_kwargs, run_job, parse_dict_with_datatypes


router = APIRouter(
    prefix="/primitive",
    tags=["primitive"],
)


@router.post("/featurize")
async def featurize(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    featurizer: Annotated[str, Body()],
    output: Annotated[str, Body()],
    dataset_column: Annotated[str, Body()],
    feat_kwargs: Annotated[Optional[Dict], Body()] = None,
    label_column: Annotated[Optional[str], Body()] = None,
) -> dict:
    """
    Submits a featurization job

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    dataset_address: str
        datastore address of dataset to featurize
    featurizer: str
        featurizer to use
    output: str
        name of the featurized dataset
    dataset_column: Optional[str]
        Column containing the input for featurizer
    feat_kwargs: Optional[Dict]
        Keyword arguments to pass to featurizer on initialization
    label_column: Optional[str]
        The target column in case this dataset is going to be used for training purposes

    Raises
    ------
    HTTPException
        If the featurizer is invalid or if the featurization fails.

    Returns
    -------
    dict
        A dictionary containing the address of the featurized dataset.
    """
    if not feat_kwargs:
        feat_kwargs = {'feat_kwargs': {}}

    if featurizer not in featurizer_map.keys():
        message = f"Invalid featurizer: {featurizer}. Use one of {list(featurizer_map.keys())}."
        raise HTTPException(status_code=404, detail=message)

    if isinstance(feat_kwargs['feat_kwargs'], str):
        feat_kwargs['feat_kwargs'] = json.loads(feat_kwargs['feat_kwargs'])

    for key, value in feat_kwargs['feat_kwargs'].items():
        if isinstance(value, str):
            if value.lower() == "true":
                feat_kwargs['feat_kwargs'][key] = True
            elif value.lower() == "false":
                feat_kwargs['feat_kwargs'][key] = False
            elif value.lower() == "none":
                feat_kwargs['feat_kwargs'][key] = None

    program: Dict = {
        'program_name': 'featurize',
        'dataset_address': dataset_address,
        'featurizer': featurizer,
        'output': output,
        'dataset_column': dataset_column,
        'feat_kwargs': feat_kwargs['feat_kwargs'],
        'label_column': label_column,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Featurization failed: {str(e)}")

    return {"featurized_file_address": str(result)}


@router.post("/train")
async def train(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    model_type: Annotated[str, Body()],
    model_name: Annotated[str, Body()],
    init_kwargs: Annotated[Optional[Dict], Body()] = None,
    train_kwargs: Annotated[Optional[Dict], Body()] = None,
) -> dict:
    """
    Submits a training job

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    dataset_address: str
        datastore address of dataset to train on
    model_type: str
        type of model to train
    model_name: str
        name of the trained model
    init_kwargs: Optional[Dict]
        Keyword arguments to pass to model on initialization
    train_kwargs: Optional[Dict]
        Keyword arguments to pass to model on training

    Raises
    ------
    HTTPException
        If the model type is invalid or if the training fails.

    Returns
    -------
    dict
        A dictionary containing the address of the trained model.
    """
    if not init_kwargs:
        init_kwargs = {}
    if not train_kwargs:
        train_kwargs = {}

    if model_type not in model_mappings.model_address_map.keys():
        message = f"Invalid model type: {model_type}. Use one of {list(model_mappings.model_address_map.keys())}."
        raise HTTPException(status_code=404, detail=message)

    if isinstance(init_kwargs, str):
        init_kwargs = json.loads(init_kwargs)

    if isinstance(train_kwargs, str):
        train_kwargs = json.loads(train_kwargs)

    if init_kwargs is not None:
        init_kwargs = parse_boolean_none_values_from_kwargs(init_kwargs)
    if train_kwargs is not None:
        train_kwargs = parse_boolean_none_values_from_kwargs(train_kwargs)

    program: Dict = {
        "program_name": "train",
        "dataset_address": dataset_address,
        "model_type": model_type,
        "model_name": model_name,
        "init_kwargs": init_kwargs,
        "train_kwargs": train_kwargs,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    return {"trained_model_address": str(result)}


@router.post("/evaluate")
async def evaluate(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    dataset_addresses: Annotated[List[str], Body()],
    model_address: Annotated[str, Body()],
    metrics: Annotated[List[str], Body()],
    output_key: Annotated[str, Body()],
    is_metric_plots: Annotated[bool, Body()] = False,
) -> dict:
    """
    Submits an evaluation job

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    dataset_addresses: List[str]
        List of dataset addresses to evaluate the model on
    model_address: str
        datastore address of the trained model
    metrics: List[str]
        List of metrics to evaluate the model with
    output_key: str
        Name of the evaluation output
    is_metric_plots: bool
        Whether plot based metric is used or not
    """

    program: Dict = {
        "program_name": "evaluate",
        "dataset_addresses": dataset_addresses,
        "model_address": model_address,
        "metrics": metrics,
        "output_key": output_key,
        "is_metric_plots": is_metric_plots,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

    return {"evaluation_result_address": str(result)}


@router.post("/infer")
async def infer(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    model_address: Annotated[str, Body()],
    data_address: Annotated[str, Body()],
    output: Annotated[str, Body()],
    dataset_column: Annotated[Optional[str], Body()] = None,
    shard_size: Annotated[Optional[int], Body()] = 8192,
    threshold: Annotated[Optional[Union[int, float]], Body()] = None,
) -> dict:
    """
    Submits an inference job

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    model_address: str
        datastore address of the trained model
    data_address: str
        datastore address of the dataset to run inference on
    output: str
        Name of the output inference results
    dataset_column: Optional[str]
        Column in the dataset to perform inference on (required for raw CSV data)
    shard_size: Optional[int]
        Shard size for the inference operation (default: 8192)
    threshold: Optional[Union[int, float]]
        Threshold for binarizing predictions (for classification tasks)

    Raises
    ------
    HTTPException
        If the model address or data address is invalid, or if the inference fails.

    Returns
    -------
    Dict or JSONResponse
        Success: {"inference_results_address": str}
        Error: JSONResponse with error message
    """
    # Convert string parameters for JSON parsing
    if isinstance(threshold, str):
        try:
            threshold = float(threshold)
        except ValueError:
            if threshold.lower() == "none":
                threshold = None
            else:
                raise HTTPException(status_code=400, detail={f"Invalid threshold value: {threshold}"})

    if isinstance(shard_size, str):
        try:
            shard_size = int(shard_size)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid shard_size value: {shard_size}")

    # Handle None values
    if dataset_column == "None":
        dataset_column = None

    # Build the program for inference
    program = {
        "program_name": "infer",
        "model_address": model_address,
        "data_address": data_address,
        "output": output,
        "dataset_column": dataset_column,
        "shard_size": shard_size,
        "threshold": threshold,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return {"inference_results_address": str(result)}


@router.post("/train-valid-test-split")
async def train_valid_test_split(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    splitter_type: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
) -> dict:
    """
    API for making train, test and validation split of data

    Parameters
    ----------
    splitter_type: str
        Type of splitter to use - `random` or `index` or `scaffold`
    dataset_address: str
        dataset to split
    frac_train: float
        fraction of train dataset
    frac_test: float
        fraction of train dataset
    frac_valid: float
        fraction of train dataset
    """

    # Build the program for Train Valid Test split
    program = {
        "program_name": "train_valid_test_split",
        "splitter_type": splitter_type,
        "dataset_address": dataset_address,
        "frac_train": frac_train,
        "frac_test": frac_test,
        "frac_valid": frac_valid,
    }

    if not math.isclose(frac_valid + frac_test + frac_train, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise HTTPException(status_code=400, detail=f"Invalid fractions: {frac_train}, {frac_test}, {frac_valid}")
    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Train valid test split failed: {str(e)}")

    return {"train_valid_test_split_results_address": result}


@router.post("/partition")
async def partition(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    n_partition: Annotated[int, Body()] = 4,
    shuffle: Annotated[bool, Body()] = False,
) -> dict:
    """
    Partition a dataset into N datastore datasets.

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    dataset_address: str
        datastore address of dataset to partition
    n_partition: int
        Number of partitions to create
    shuffle: bool
        Whether to shuffle before partitioning
    """
    program: Dict = {
        "program_name": "partition",
        "dataset_address": dataset_address,
        "n_partition": n_partition,
        "shuffle": shuffle,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Partition failed: {str(e)}")

    return {"partitioned_dataset_addresses": result}


@router.post("/generate_pose")
async def docking_generate_pose(
    profile_name: str,
    project_name: str,
    protein_address: str,
    ligand_address: str,
    output: str,
    exhaustiveness: int = 10,
    num_modes: int = 9,
) -> dict:
    """
    Generate VINA molecular docking poses.

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    protein_address: str
        DeepChem address of the protein PDB file
    ligand_address: str
        DeepChem address of the ligand file (PDB/SDF)
    output: str
        Output name for the docking results
    exhaustiveness: int
        Vina exhaustiveness parameter (default: 10)
    num_modes: int
        Number of binding modes to generate (default: 9)

    Returns
    -------
    dict
        Dictionary containing the address of the docking results
    """

    program = {
        'program_name': 'generate_pose',
        'protein_address': protein_address,
        'ligand_address': ligand_address,
        'output': output,
        'exhaustiveness': exhaustiveness,
        'num_modes': num_modes,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VINA docking failed: {str(e)}")

    return {"docking_results_address": str(result)}


@router.post("/fep/calculate_rbfe")
async def relative_binding_free_energy(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    solvent: Dict[Any, Any],
    ligands_sdf_address: str,
    cleaned_protein_pdb_address: str,
    overridden_rbfe_settings: Dict[Any, Any],
    radial_network_central_ligand: Optional[str],
    dry_run: bool = False,
    run_edges_in_parallel: bool = False,
    network_type: Optional[str] = "MINIMAL_SPANNING",
    scorer_type: Optional[str] = 'LOMAP',
    output_key: Optional[str] = "output_key",
) -> dict:
    """API Route for submitting relative binding free energy calculation jobs.

    Parameters
    ----------
    solvent : Dict
        Solvent component dictionary, by default None
    ligands_sdf_address : str, optional
        DeepChem datastore address of the ligands.sdf file, by default None
    reference_ligand : str, optional
        The name of the reference ligand, for example "benzene", "toluene" etc.
        This MUST be one of the ligands present in the ligands .SDF file, by default None
    cleaned_protein_pdb_address : str, optional
        Deepchem datastore address of a protein.pdb file, by default None
    overridden_rbfe_settings : Dict, optional
        The serialized JSON representation of the RelativeHybridTopologyProtocolSettings object.
        Only the settings that are to be overridden should be included in this JSON string, by default None
    dry_run : bool, optional
        Whether to run the RBFE calculation in dry run mode, by default False
    run_edges_in_parallel : bool, optional
        Whether to run the edges in parallel, by default False
    network_type : str, optional
        The type of network to use, for example RADIAL or MINIMAL_SPANNING. By default, MINIMAL_SPANNING.
    scorer_type : str, optional
        The type of scorer to use, for example LOMAP. By default, LOMAP.
    output_key : str, optional
        The output directory for storing the results, by default "output_key"

    Returns
    -------
    dict
        Dictionary containing the address of the relative binding free energy results.
    """
    from deepchem_server.core.primitives.fep.rbfe.utils.constants import NetworkPlanningConstants

    if overridden_rbfe_settings is not None:
        try:
            for key, value in overridden_rbfe_settings.items():
                if key == 'protocol_repeats':
                    if value and len(value) < 10000:
                        parsed_value = ast.literal_eval(value)
                        overridden_rbfe_settings[key] = parsed_value
                        continue
                    else:
                        raise HTTPException(status_code=422, \
                                            detail="Invalid value for 'protocol_repeats' setting.")
                overridden_rbfe_settings[key] = parse_dict_with_datatypes(value)
        except (ValueError, SyntaxError):
            raise HTTPException(status_code=422, detail="Could not parse Settings")

    if solvent is None:
        raise HTTPException(status_code=400, detail="Solvent is required")
    try:
        solvent = parse_dict_with_datatypes(solvent)
    except (ValueError, SyntaxError):
        raise HTTPException(status_code=422, detail="Could not parse Solvent Settings")

    # Validate the network type
    try:
        network_type = network_type.upper()  # type: ignore
        NetworkPlanningConstants.PerturbationNetworkType(network_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=  # noqa: E251
            f"Invalid network type: {network_type}. Must be one of {[NetworkPlanningConstants.PerturbationNetworkType._member_names_]}"
        )

    # Validate the scorer type
    try:
        if not scorer_type or scorer_type == 'None':
            scorer_type = None
        else:
            NetworkPlanningConstants.ScorerType(scorer_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=  # noqa: E251
            f"Invalid scorer type: {scorer_type}. Must be one of {[*NetworkPlanningConstants.ScorerType._member_names_, None]}"
        )

    # Parse reference_ligand
    if isinstance(radial_network_central_ligand, str):
        try:
            radial_network_central_ligand = ast.literal_eval(radial_network_central_ligand)
        except (ValueError, SyntaxError):
            if radial_network_central_ligand == '':
                radial_network_central_ligand = None

    program: Dict = {
        "program_name": "relative_binding_free_energy",
        "ligands_sdf_address": ligands_sdf_address,
        "cleaned_protein_pdb_address": cleaned_protein_pdb_address,
        "network_type": network_type,
        "scorer_type": scorer_type,
        "solvent": solvent,
        "overridden_rbfe_settings": overridden_rbfe_settings,
        "dry_run": dry_run,
        "radial_network_central_ligand": radial_network_central_ligand,
        "output_key": output_key,
    }

    if run_edges_in_parallel:
        raise NotImplementedError("Parallel edge execution is not implemented")
    else:
        try:
            result = run_job(profile_name=profile_name, project_name=project_name, program=program)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Relative binding free energy calculation failed: {str(e)}")
    return {"relative_binding_free_energy_results_address": str(result)}


@router.post("/del/denoise")
async def del_denoise(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    output_key: Annotated[str, Body()],
    strategy: Annotated[str, Body()] = 'unified',
    control_cols: Annotated[Optional[List[str]], Body()] = None,
    target_cols: Annotated[Optional[List[str]], Body()] = None,
    add_hit_labels: Annotated[bool, Body()] = False,
    hit_percentile: Annotated[float, Body()] = 90.0,
    alpha: Annotated[float, Body()] = 0.05,
    drop_duplicates: Annotated[bool, Body()] = True,
    use_disynthon_pairs: Annotated[bool, Body()] = False,
    smiles_cols: Annotated[Optional[List[str]], Body()] = None,
    aggregate_operation: Annotated[str, Body()] = 'sum',
    min_count_threshold: Annotated[int, Body()] = 0,
) -> dict:
    """Denoise DEL sequencing data by computing enrichment scores.

    Optionally collapses trisynthon data into disynthon pairs before
    enrichment scoring when use_disynthon_pairs=True.

    Parameters
    ----------
    profile_name : str
        Name of the Profile where the job is run.
    project_name : str
        Name of the Project where the job is run.
    dataset_address : str
        Datastore address of the raw DEL CSV.
    output_key : str
        Name for the denoised output dataset.
    strategy : str
        'unified' (Poisson CI ratio) or 'non_unified' (z-score).
    control_cols : List[str], optional
        Control replicate count column names.
    target_cols : List[str], optional
        Target replicate count column names.
    add_hit_labels : bool
        Whether to add binary hit label columns.
    hit_percentile : float
        Percentile threshold for hit classification (0-100).
    alpha : float
        Significance level for Poisson CI (unified only).
    drop_duplicates : bool
        Whether to drop duplicate SMILES rows.
    use_disynthon_pairs : bool
        If True, collapse trisynthon data into disynthon pairs before scoring.
    smiles_cols : List[str], optional
        Three synthon SMILES column names. Required when use_disynthon_pairs=True.
    aggregate_operation : str
        Aggregation function for disynthon deduplication: 'sum' or 'mean'.
    min_count_threshold : int
        Minimum total count per disynthon to retain.

    Returns
    -------
    dict
        Dictionary containing the denoised dataset address.
    """
    if strategy not in ('unified', 'non_unified'):
        raise HTTPException(status_code=422,
                            detail=f"Invalid strategy '{strategy}'. Must be 'unified' or 'non_unified'.")
    if not 0 < hit_percentile < 100:
        raise HTTPException(status_code=422, detail=f"hit_percentile must be between 0 and 100, got {hit_percentile}")
    if not 0 < alpha < 1:
        raise HTTPException(status_code=422, detail=f"alpha must be between 0 and 1, got {alpha}")
    if use_disynthon_pairs and smiles_cols is not None and len(smiles_cols) != 3:
        raise HTTPException(status_code=422,
                            detail=f"smiles_cols must contain exactly 3 column names, got {len(smiles_cols)}")
    if aggregate_operation not in ('sum', 'mean'):
        raise HTTPException(status_code=422,
                            detail=f"aggregate_operation must be 'sum' or 'mean', got '{aggregate_operation}'")

    program: Dict = {
        "program_name": "del_denoise",
        "dataset_address": dataset_address,
        "output_key": output_key,
        "strategy": strategy,
        "control_cols": control_cols,
        "target_cols": target_cols,
        "add_hit_labels": add_hit_labels,
        "hit_percentile": hit_percentile,
        "alpha": alpha,
        "drop_duplicates": drop_duplicates,
        "use_disynthon_pairs": use_disynthon_pairs,
        "smiles_cols": smiles_cols,
        "aggregate_operation": aggregate_operation,
        "min_count_threshold": min_count_threshold,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DEL denoising failed: {str(e)}")

    return {"denoised_dataset_address": str(result)}


@router.post("/fep/collate_rbfe_results")
async def collate_rbfe_results(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    result_files_addresses: Annotated[List[str], Body()],
    output_file_name: Annotated[str, Body()],
    reference_ligand: Annotated[Optional[str], Body()] = None,
    reference_ligand_dg_value: Annotated[Optional[str], Body()] = None,
    reference_ligand_dg_value_uncertainty: Annotated[Optional[str], Body()] = None,
) -> dict:
    """API Route for submitting relative collate RBFE results jobs.

    Parameters
    ----------
    result_files_addresses : List[str], optional
        The list of DeepChem addressed of RBFE results files to be processed, by default Body()
    reference_ligand : Union[None, str], optional
        The reference ligand whose DG value is know, by default Body(None)
    reference_ligand_dg_value : Union[None, str], optional
        The DG value of the reference ligand, by default Body(None)
    reference_ligand_dg_value_uncertainty : Union[None, str], optional
        The uncertainty in the DG value of the reference ligand, by default Body(None)
    output_file_name : str, optional
        Name of the generated output file, by default Body()

    Returns
    -------
    dict
        Dictionary containing the address of the collated relative binding free energy results.
    """
    import pint
    from deepchem_server.core.primitives.fep.rbfe.collate_rbfe_results import (
        process_input_files,
        get_ligands_from_results,
    )

    try:
        if reference_ligand_dg_value:
            pint.Quantity(reference_ligand_dg_value)  # type: ignore
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=  # noqa: E251
            "Please enter a valid PlainQuantity string for reference ligand dg value. For example, 2.0 kilocalorie/mol",
        )

    try:
        if reference_ligand_dg_value_uncertainty:
            pint.Quantity(reference_ligand_dg_value_uncertainty)  # type: ignore
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=  # noqa: E251
            "Please enter a valid PlainQuantity string for reference ligand dg value uncertainty. For example, 0.02 kilocalorie/mol",
        )

    simulation_results = process_input_files(result_files_addresses)
    ligands = get_ligands_from_results(simulation_results)

    if not ligands:
        raise HTTPException(status_code=422, detail="No ligands found in given results")

    if reference_ligand and reference_ligand not in ligands:
        raise HTTPException(
            status_code=422,
            detail=  # noqa: E251
            f"Reference ligand {reference_ligand} not found in given results",
        )

    # The DeepChem language does not support lists with single quotes, so we replace all single quotes with double quotes.
    result_files_string_representation = (str(result_files_addresses)).replace("'", '"')

    program: Dict = {
        "program_name": "collate_rbfe_results",
        "result_files_addresses": result_files_string_representation,
        "reference_ligand": reference_ligand,
        "reference_ligand_dg_value": reference_ligand_dg_value,
        "reference_ligand_dg_value_uncertainty": reference_ligand_dg_value_uncertainty,
        "output_file_name": output_file_name,
    }
    try:
        result = run_job(
            profile_name=profile_name,
            project_name=project_name,
            program=program,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collate relative binding free energy results failed: {str(e)}")

    return {"collate_relative_binding_free_energy_results_address": str(result)}


@router.post("/transform")
async def apply_transform(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    transform_type: Annotated[str, Body()],
    column_name: Annotated[str, Body()],
    new_column_name: Annotated[str, Body()],
    output_key: Annotated[str, Body()],
) -> dict:
    """
    Submits a transform job.

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    dataset_address: str
        datastore address of dataset to transform
    transform_type: str
        transform type to use (e.g., 'log' or 'norm')
    column_name: str
        name of the column to transform
    new_column_name: str
        name of the resulting column
    output_key: str
        name of the output transformed dataset
    """
    program: Dict = {
        'program_name': 'transform',
        'dataset_address': dataset_address,
        'transform_type': transform_type,
        'column_name': column_name,
        'new_column_name': new_column_name,
        'output_key': output_key
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transform failed: {str(e)}")

    return {"transformed_file_address": str(result)}


@router.post("/cluster")
async def apply_cluster(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    num_clusters: Annotated[int, Body()],
    column: Annotated[str, Body()],
    output: Annotated[str, Body()],
) -> dict:
    """
    Submits a clustering job.

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    dataset_address: str
        datastore address of dataset to cluster
    num_clusters: int
        Number of clusters (k)
    column: str
        The name of SMILES column to cluster
    output: str
        Output file prefix
    """
    program: Dict = {
        'program_name': 'cluster',
        'dataset_address': dataset_address,
        'num_clusters': num_clusters,
        'column': column,
        'output': output
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

    # result will be a tuple of (prediction_address, cluster_center_address)
    # The current HTTP response model expects a dict.
    return {
        "prediction_address": result[0],
        "cluster_center_address": result[1],
    }


@router.post("/hyperparam-opt")
async def apply_hyperparam_opt(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    model_type: Annotated[str, Body()],
    train_address: Annotated[str, Body()],
    valid_address: Annotated[str, Body()],
    hyperparams: Annotated[Dict, Body()],
    output_prefix: Annotated[str, Body()],
    algorithm: Annotated[Optional[str], Body()] = "grid",
    metric: Annotated[str, Body()] = "pearson_r2_score",
    nb_epoch: Annotated[int, Body()] = 10,
) -> dict:
    """
    Submits a hyperparameter optimization job.

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    model_type: str
        Model type name recognizable by deepchem_server
    train_address: str
        datastore address of training dataset
    valid_address: str
        datastore address of validation dataset
    hyperparams: Dict
        dictionary of hyperparameter names mapped to lists of values to search
    output_prefix: str
        Output prefix for best model and search results
    algorithm: Optional[str]
        Hyperparameter selection algorithm (Default: grid)
    metric: str
        Evaluation metric name
    nb_epoch: int
        Number of epochs to train each candidate model
    """
    program: Dict = {
        'program_name': 'hyperparam_opt',
        'model_type': model_type,
        'train_address': train_address,
        'valid_address': valid_address,
        'hyperparams': hyperparams,
        'output_prefix': output_prefix,
        'algorithm': algorithm,
        'metric': metric,
        'nb_epoch': nb_epoch
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hyperparameter optimization failed: {str(e)}")

    # result will be a tuple: (model_address, output_address_best_hyperparams, output_address)
    return {
        "model_address": result[0],
        "best_hyperparams_address": result[1],
        "all_results_address": result[2],
    }
