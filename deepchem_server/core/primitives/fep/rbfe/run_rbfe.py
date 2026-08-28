# Patch Pydantic V2 and V1 to prevent OpenFE validation conflicts
try:
    import pydantic
    if hasattr(pydantic, "main") and hasattr(pydantic.main, "ModelMetaclass"):
        _orig_new = pydantic.main.ModelMetaclass.__new__

        def _patched_new(mcs, name, bases, namespace, **kwargs):
            if "reassign_velocities" in namespace:
                annotations = namespace.setdefault("__annotations__", {})
                if "reassign_velocities" not in annotations:
                    annotations["reassign_velocities"] = bool
            return _orig_new(mcs, name, bases, namespace, **kwargs)

        pydantic.main.ModelMetaclass.__new__ = _patched_new

    import pydantic.class_validators as cv
    _orig_v1 = cv.validator

    def _patched_v1(*args, **kwargs):
        kwargs['allow_reuse'] = True
        return _orig_v1(*args, **kwargs)

    cv.validator = _patched_v1
    if hasattr(pydantic, "validator"):
        pydantic.validator = _patched_v1
except Exception:
    pass

try:
    import pydantic.v1.class_validators as cv_v2
    _orig_v2 = cv_v2.validator

    def _patched_v2(*args, **kwargs):
        kwargs['allow_reuse'] = True
        return _orig_v2(*args, **kwargs)

    cv_v2.validator = _patched_v2
    import pydantic.v1 as pv1
    if hasattr(pv1, "validator"):
        pv1.validator = _patched_v2
except Exception:
    pass

import ast
import json
import logging
import os
from typing import Dict, Optional

from deepchem_server.core.common import config
from deepchem_server.core.common.address import DeepchemAddress
from deepchem_server.core.common.cards import DataCard
from deepchem_server.core.primitives.fep.rbfe import system_setup
from deepchem_server.core.primitives.fep.rbfe.utils.constants import NetworkPlanningConstants
from deepchem_server.core.primitives.fep.rbfe.utils.rbfe_utils import RBFESettingsUtils, SolventComponentUtils


def run_rbfe(ligands_sdf_address: str,
             cleaned_protein_pdb_address: str,
             network_type: str,
             scorer_type: Optional[str],
             solvent_json: str,
             overridden_rbfe_settings: str,
             dry_run: bool = False,
             radial_network_central_ligand: Optional[str] = None,
             output_key: Optional[str] = None) -> list[str]:
    """A FEP utility to run RBFE calculations for a protein-solvent-ligand system, for a list of ligands.

    If run as an Array Job, Each edge in the perturbation network is run as an independent job within the same AWS Batch array job.
    This allows us to group the different edges in the perturbation network into a single AWS Batch array job.

    For example, if a calculation involves 3 edges to be run, then the AWS Batch array will have 3 jobs.

    For each job within an array job, the job command stays the same. Since each job within an array job is for a different edge, we need to determine which edge to run for each job.
    This edge identification is achieved by the use of the AWS_BATCH_JOB_ARRAY_INDEX environment variable.
    The AWS_BATCH_JOB_ARRAY_INDEX  is set by AWS Batch runner, and is available as an environment variable in the job environment.

    If the AWS_BATCH_JOB_ARRAY_INDEX environment variable is not set, it indicates the the job is being run outside of AWS.
    In such a case, all the edges of the perturbation network are executed serially, one after another.

    Upon successful execution, the result of the RBFE calculation is written to a JSON file, which is then uploaded to the user's project directory.
    Each edge in the perturbation network generates a corresponding JSON file. If the perturbation network has N edges, then N files are generated

    The format of the output file name is: rbfe-{componentA}-{componentB}.json where componentA and componentB are the names of the components in the edge.

    Since the number of the resultant JSON files can be large, it is recommended
    to use a separate project directory for each RBFE calculation.

    Example
    -------
    >>> from deepchem_server.core.primitives.fep.rbfe.run_rbfe import run_rbfe
    >>> ligands_sdf_address = "deepchem://test/test-starter/ligands.sdf"
    >>> cleaned_protein_pdb_address = "deepchem://test/test-starter/cleaned-protein.pdb"
    >>> network_type = "RADIAL"
    >>> scorer_type = "LOMAP"
    >>> solvent_json = '{"positive_ion": "Na", "negative_ion": "Cl", "ion_concentration": "0.15 mol/L", "neutralize": true}'
    >>> overridden_rbfe_settings = '{"simulation_settings" : {"equilibration_length" : 2 ns}}'
    >>> run_rbfe(ligands_sdf_address, cleaned_protein_pdb_address, network_type, scorer_type, solvent_json, overridden_rbfe_settings, False)

    Parameters
    ----------
    ligands_sdf_address : str
        The deepchem datastore address of the ligands .SDF file.
    cleaned_protein_pdb_address : str
        The deepchem datastore address of the cleaned protein .PDB file.
    network_type: str
        The type of the Perturbation Network to be generated.
        For example, RADIAL, MINIMAL_SPANNING
    scorer_type : str, optional
        The type of the Scorer to be used while generating the Perturbation Network.
        For example, LOMAP
    solvent : str
        The serialized JSON representation of the SolventComponent object.
    overridden_rbfe_settings : str
        The serialized JSON representation of the RelativeHybridTopologyProtocolSettings object.
        Only the settings that are to be overridden should be included in this JSON string.
    dry_run : bool, optional
        Whether to run the RBFE calculation in dry run mode.
    radial_network_central_ligand : str
        The name of the reference ligand, for example "benzene", "toluene" etc.
        This MUST be one of the ligands present in the ligands .SDF file.
        Needs to be specified for Radial Networks only.
    output_key : str, optional
        The key to be used for storing the results.
        If not specified, the results are stored in the default working directory.
    """

    logger = logging.getLogger(__name__)

    # The AWS_BATCH_JOB_ARRAY_INDEX environment variable is set by the AWS Runner when running batch array jobs.
    try:
        AWS_BATCH_JOB_ARRAY_INDEX = int(os.getenv('AWS_BATCH_JOB_ARRAY_INDEX'))  # type: ignore
    except TypeError:
        AWS_BATCH_JOB_ARRAY_INDEX = None

    if not radial_network_central_ligand or radial_network_central_ligand == 'None':
        radial_network_central_ligand = None

    # Validate the scorer type
    if not scorer_type or scorer_type == 'None':
        scorer_type = None
    else:
        try:
            NetworkPlanningConstants.ScorerType(scorer_type)
        except ValueError:
            raise ValueError(
                f"Invalid scorer type: {scorer_type}. Must be one of {[*NetworkPlanningConstants.ScorerType._member_names_, None]}"
            )

    # Validate the network type
    try:
        NetworkPlanningConstants.PerturbationNetworkType(network_type)
    except ValueError:
        raise ValueError(
            f"Invalid network type: {network_type}. Must be one of {[NetworkPlanningConstants.PerturbationNetworkType._member_names_]}"
        )

    if network_type == NetworkPlanningConstants.PerturbationNetworkType.RADIAL.name:
        if radial_network_central_ligand is None:
            logger.warn(
                "Central ligand not specified for radial network type. Automatically choosing the most apt ligand as the central ligand based on similarity scores."
            )
    else:
        if radial_network_central_ligand is not None:
            logger.warn(
                f"Central ligand specified for non-radial network type. Ignoring the central ligand parameter - {radial_network_central_ligand}"
            )
        if scorer_type is None:
            raise Exception(
                f"scorer_type cannot be None for Non-Radial networks. Please select a valid scorer type from {NetworkPlanningConstants.ScorerType._member_names_}"
            )

    # Load the ligands and the protein.
    ligand_mols = system_setup.load_ligands(ligands_sdf_address)
    protein = system_setup.load_cleaned_protein(cleaned_protein_pdb_address)

    # Define the transformation network.
    network = system_setup.get_perturbation_network(ligand_mols,
                                                    NetworkPlanningConstants.PerturbationNetworkType(network_type),
                                                    NetworkPlanningConstants.ScorerType(scorer_type),
                                                    central_ligand=radial_network_central_ligand)

    # Define the RBFE settings and the solvent.

    if isinstance(overridden_rbfe_settings, str):
        overridden_rbfe_settings = ast.literal_eval(overridden_rbfe_settings)

    if isinstance(overridden_rbfe_settings, Dict) and "protocol_repeats" in overridden_rbfe_settings:
        protocol_repeats: int = ast.literal_eval(overridden_rbfe_settings['protocol_repeats'])
        del overridden_rbfe_settings['protocol_repeats']

        rbfe_settings = RBFESettingsUtils.loads(json.dumps(overridden_rbfe_settings))
        rbfe_settings.protocol_repeats = protocol_repeats
    else:
        rbfe_settings = RBFESettingsUtils.loads(json.dumps(overridden_rbfe_settings))

    if isinstance(solvent_json, str):
        solvent_json = ast.literal_eval(solvent_json)
    solvent = SolventComponentUtils.loads(json.dumps(solvent_json))

    # Parse dry run option if it a string
    if isinstance(dry_run, str):
        dry_run = ast.literal_eval(dry_run)
    assert isinstance(dry_run, bool)

    # Setup the transformations.
    runnable_edges, rbfe_transform = system_setup.setup_transformations(network, solvent, protein, rbfe_settings)

    # If the AWS_BATCH_JOB_ARRAY_INDEX environment variable is set, then only run the edge corresponding to the index. (Other edges will be handled by sibling child jobs)
    # If the environment variable is not set, then run all the edges serially.
    if AWS_BATCH_JOB_ARRAY_INDEX is not None:
        runnable_edges = [runnable_edges[AWS_BATCH_JOB_ARRAY_INDEX]]
        logger.info(f"Found AWS_BATCH_JOB_ARRAY_INDEX in the environment. Running edge {AWS_BATCH_JOB_ARRAY_INDEX}")
    else:
        logger.warning(f"Could not find AWS_BATCH_JOB_ARRAY_INDEX. Running all {len(runnable_edges)} edges serially.")
    result_datastore_address_list = []

    for i, edge in enumerate(runnable_edges):

        if dry_run:
            result = system_setup.dry_run_edge(edge)
        else:
            result = system_setup.run_edge(edge, rbfe_transform)

        logger.info(f"RESULTS: {result.__dict__}")

        # Write the result to a temporary file.
        with open(f'tmp_{i+1}.json', 'w') as f:
            f.write(json.dumps(result.__dict__, indent=2))  # type: ignore

        # Upload the result to the datastore.

        result_datacard = DataCard(
            address='',
            file_type='json',
            data_type='text/plain',
        )

        RESULT_FILE_NAME = f"{DeepchemAddress.get_parent_key(ligands_sdf_address)}rbfe-{edge.componentA.name}-{edge.componentB.name}.json"

        if output_key:
            RESULT_FILE_NAME = f"{output_key}/{RESULT_FILE_NAME}"

        logger.info(f"Saving results in: {RESULT_FILE_NAME}")

        datastore = config.get_datastore()
        if datastore is None:
            raise ValueError("Datastore not set")
        result_datastore_address = datastore.upload_data(RESULT_FILE_NAME, f"tmp_{i+1}.json", result_datacard)

        logger.info(f"Results saved at address: {result_datastore_address}")

        # Append the result datastore address to the list of result datastore addresses.
        result_datastore_address_list.append(result_datastore_address)

    return result_datastore_address_list
