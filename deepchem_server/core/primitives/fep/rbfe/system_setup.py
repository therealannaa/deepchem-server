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

import os
from pathlib import Path
import tempfile
from typing import Optional, Tuple, Type

from gufe.ligandnetwork import LigandNetwork  # type: ignore
from gufe.protocols import execute_DAG  # type: ignore
import numpy as np  # type: ignore
from openfe import (  # type: ignore
    ChemicalSystem, lomap_scorers, ProteinComponent, SmallMoleculeComponent, SolventComponent,
)
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol  # type: ignore
from openfe.protocols.openmm_rfe.equil_rfe_settings import RelativeHybridTopologyProtocolSettings  # type: ignore
from openfe.setup import LomapAtomMapper  # type: ignore
from openfe.setup.ligand_network_planning import (  # type: ignore
    generate_maximal_network, generate_minimal_spanning_network, generate_radial_network,
)
from openff.units import unit  # type: ignore
import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore

from deepchem_server.core.common import config
from deepchem_server.core.primitives.fep.rbfe.data_domain_classes.EdgeSimulationResult import EdgeSimulationResult
from deepchem_server.core.primitives.fep.rbfe.data_domain_classes.RunnableEdge import RunnableEdge
from deepchem_server.core.primitives.fep.rbfe.utils.constants import (
    DEBUG,
    FAILURE,
    LIGAND,
    NetworkPlanningConstants,
    PROTEIN,
    SAMPLER,
    SOLVENT,
    SUCCESS,
)


def load_ligands(sdf_datastore_address: str) -> list[SmallMoleculeComponent]:
    """Loads a .sdf file from the datastore.

    This function loads a ligands.sdf file from the datastore and returns a corresponding list of
    SmallMoleculeComponent objects.

    Parameters
    ----------
    sdf_datastore_address: str
      The address of ligand SDF file dataset to load.

    Returns
    -------
    ligand_mols: list[SmallMoleculeComponent]
        A list of SmallMoleculeComponent objects, where each object represents a ligand
    """
    datastore = config.get_datastore()

    if datastore is None:
        raise Exception("Datastore not set.")

    # Make temporary file for loaded data
    tempdir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(tempdir.name, "temp.sdf")
    datastore.download_object(sdf_datastore_address, temp_file_path)

    ligands = Chem.SDMolSupplier(temp_file_path, removeHs=False)

    if ligands is None:
        raise Exception("Ligands input is invalid.")

    ligand_mols = [SmallMoleculeComponent(mol) for mol in ligands]

    return ligand_mols


def calculate_lomap_similarity(ligand_mols: list[SmallMoleculeComponent]) -> Type[np.ndarray]:
    """Calculates the LOMAP similarity among all possible pairs of ligands in a list of ligands.

    This can be used to determine the reference(central) ligand for radial ligand network generation.

    Parameters
    ----------
    ligand_mols : list[SmallMoleculeComponent]
        List of OpenFE SmallMoleculeComponent ligands.

    Returns
    -------
    similarity_array : Type[numpy.ndarray]
        2D array of LOMAP similarity scores where a score of 1 indicates identity and
        0 indicates no alchemical similarity.
    """
    similarity_array = np.zeros((len(ligand_mols), len(ligand_mols)))

    # Calculate similarity for upper triangle, without diagonal
    for row in range(0, len(ligand_mols) - 1):
        for col in range(row + 1, len(ligand_mols)):
            m1 = ligand_mols[row]
            m2 = ligand_mols[col]
            mapper = LomapAtomMapper(threed=True)
            mapping = next(mapper.suggest_mappings(m1, m2))
            score = 1 - lomap_scorers.default_lomap_score(mapping)
            similarity_array[row, col] = score

    # Fill in diagonal
    np.fill_diagonal(similarity_array, 0.5)

    # Symmetrize similarity array
    similarity_array = similarity_array + similarity_array.T

    return similarity_array  # type: ignore


def get_reference_ligand(ligand_mols: list[SmallMoleculeComponent]) -> int:
    """Returns the index of the Ligand in ligand_mols, most suitable to
    be the reference ligand for the radial network.

    A reference ligand is one which is at the center of the radial network.
    This is calculated by finding the ligand with the highest sum of similarity scores.

    Parameters
    ----------
    similarity_array : Type[numpy.ndarray]
        2D array of LOMAP similarity scores where a score of 1 indicates identity and
        0 indicates no alchemical similarity.
    ligand_mols : list[SmallMoleculeComponent]
        List of OpenFE SmallMoleculeComponent ligands for simulation.

    Returns
    -------
    ref_idx : int
        Index of the reference ligand in ligand_mols.
    """

    # Calculate the similarity array.
    similarity_array = calculate_lomap_similarity(ligand_mols)

    df_summed = pd.DataFrame(
        np.sum(similarity_array, axis=1),  # type: ignore
        columns=["sum"])

    # Find the index of the maximum value in summed df.
    max_sum_id = df_summed.idxmax()

    # Get ligand name at max sum as string.
    ref_idx = max_sum_id[0]

    return ref_idx


def get_perturbation_network(ligand_mols: list[SmallMoleculeComponent],
                             network_type: NetworkPlanningConstants.PerturbationNetworkType,
                             scorer: Optional[NetworkPlanningConstants.ScorerType] = None,
                             **kwargs) -> Type[LigandNetwork]:
    """Generates a perturbation network for the given ligands.

    Parameters
    ----------
    ligand_mols : list[SmallMoleculeComponent]
        List of OpenFE SmallMoleculeComponent ligands.
    network_type : PerturbationNetworkType
        The type of network to be generated.
    scorer : Optional[ScorerType], optional
        The Scorer to be used, by default None

    Returns
    -------
    network: Type[LigandNetwork]
        A LigandNetwork contains a list of edges and nodes where
        each edge in the network is a planned transformation (simulation), and
        each node is a ligand.

        The edges can be run in parallel, and their execution is independent of other edges in the network.
    """

    assert ligand_mols is not None, "ligand_mols cannot be None."

    assert network_type is not None, "network_type cannot be None"

    if network_type in [
            NetworkPlanningConstants.PerturbationNetworkType.MAXIMAL,
            NetworkPlanningConstants.PerturbationNetworkType.MINIMAL_SPANNING
    ]:
        assert scorer is not None, "A Scorer must be specified for Maximal and Minimal Spanning Networks"

    network_planner_map = {
        NetworkPlanningConstants.PerturbationNetworkType.RADIAL: generate_radial_network,
        NetworkPlanningConstants.PerturbationNetworkType.MINIMAL_SPANNING: generate_minimal_spanning_network,
        NetworkPlanningConstants.PerturbationNetworkType.MAXIMAL: generate_maximal_network
    }

    scorer_map = {
        NetworkPlanningConstants.ScorerType.LOMAP: lomap_scorers.default_lomap_score,
    }

    network_planner_params = {
        "ligands": ligand_mols,
        "mappers": [LomapAtomMapper(threed=True, element_change=False)],
        "scorer": scorer_map[scorer] if scorer is not None else None
    }

    if network_type == NetworkPlanningConstants.PerturbationNetworkType.RADIAL:

        central_ligand = kwargs.get('central_ligand', None)

        ref_idx = get_reference_ligand_index(ligand_mols, central_ligand)

        network_planner_params["ligands"] = ligand_mols[:ref_idx] + ligand_mols[ref_idx + 1:]  # yapf: disable
        network_planner_params['central_ligand'] = ligand_mols[ref_idx]

    network = network_planner_map[network_type](**network_planner_params)

    return network


def get_reference_ligand_index(ligand_mols: list[SmallMoleculeComponent],
                               reference_ligand: Optional[str] = None) -> int:
    """Returns the index of the reference Ligand in ligand_mols.

    For radial networks, a central ligand from which all transformations originate must be specified.
    If specified, the reference_ligand argument (str) must match an entry in the ligand_mols list, and is used as is.
    Otherwise, the most suitable ligand is calculated to be the reference ligand.
    This function returns the index of the reference ligand in the ligand_mols list once the reference ligand is determined.

    Parameters
    ----------
    ligand_mols : list[SmallMoleculeComponent]
        The list of OpenFE SmallMoleculeComponent ligands.
    reference_ligand : Optional[str], optional
        The name of the reference ligand, by default None

    Returns
    -------
    index: int
        The index of the reference ligand in the ligand_mols list.

    Raises
    ------
    Exception
        If the reference_ligand argument (str) does not match an entry in the ligand_mols list.
    """

    # Calculate the reference ligand if it was not specified:
    ref_idx = None

    if reference_ligand is None:
        ref_idx = get_reference_ligand(ligand_mols)

    # The reference ligand was specified:
    elif reference_ligand is not None and isinstance(reference_ligand, str) is True:
        names = [str(i.name) for i in ligand_mols]
        try:
            ref_idx = [names.index(i) for i in names if (reference_ligand == i)][0]
        except IndexError:
            raise Exception(
                f"The reference_ligand argument (str) must match an entry in {[str(i.name) for i in ligand_mols]}.")

    return ref_idx  # type: ignore


def load_cleaned_protein(pdb_datastore_address: str) -> Type[ProteinComponent]:
    """Loads a cleaned .pdb file from datastore as a ProteinComponent

    Parameters
    ----------
    PDB : str
        Datastore address of the Protein file cleaned with PDBFixer.
        The protein must be oriented correctly with respect to input ligands.

    Returns
    -------
    protein : Type[ProteinComponent]
        The protein at the given address loaded as a ProteinComponent.
    """
    try:
        tempdir = tempfile.TemporaryDirectory()
        temp_pdb_address = os.path.join(tempdir.name, "temp_protein.pdb")
        datastore = config.get_datastore()
        if datastore is None:
            raise Exception("Datastore not set.")
        datastore.download_object(pdb_datastore_address, temp_pdb_address)
    except:  # noqa
        raise Exception("Protein input is invalid.")

    protein = ProteinComponent.from_pdb_file(temp_pdb_address)

    return protein


def get_default_RBFE_simulation_settings() -> Type[RelativeHybridTopologyProtocolSettings]:
    """Returns the default simulation settings for RBFE.

    Returns
    -------
    rbfe_settings : Type[RelativeHybridTopologyProtocolSettings]
        A RelativeHybridTopologyProtocolSettings object initialized with the default industry standard RBFE settings.
    """
    rbfe_settings = RelativeHybridTopologyProtocol.default_settings()

    # Set any default simulation settings here.
    rbfe_settings.simulation_settings.equilibration_length = 2 * unit.nanosecond
    rbfe_settings.simulation_settings.production_length = 10 * unit.nanosecond
    rbfe_settings.simulation_settings.real_time_analysis_interval = None

    return rbfe_settings


def setup_transformations(
    network: Type[LigandNetwork],
    solvent: Type[SolventComponent],
    protein: Type[ProteinComponent],
    rbfe_settings: Type[RelativeHybridTopologyProtocolSettings],
) -> Tuple[list[RunnableEdge], Type[RelativeHybridTopologyProtocol]]:
    """Constructs the chemical systems and other components necessary to run the simulation for the given network.

    Parameters
    ----------
    network : Type[LigandNetwork]
        Each edge in the network is a planned transformation (simulation), and each node is a ligand.
    solvent : Type[SolventComponent]
        The solvent conditions of the simulation.
    protein : Type[ProteinComponent]
        The protein to which the ligands bind for the simulation.
    rbfe_settings: Type[RelativeHybridTopologyProtocolSettings]
        The simulation settings specify the conditions under which the simulations are executed.
        This includes various parameters of the simulation, such as the temperature and pressure.


    Returns
    -------
    runnable_edges : list[RunnableEdge]
        A list of RunnableEdge objects containing
            - componentA
            - componentB
            - solvent_dag
            - complex_dag
        for each RunnableObject.
    rbfe_transform : Type[RelativeHybridTopologyProtocol]
        A RelativeHybridTopologyProtocol instance used to gather the results generated from executing the solvent and the complex Dags.
    """
    # Initialize RBFE simulation settings.
    if (rbfe_settings is None):
        rbfe_settings = get_default_RBFE_simulation_settings()

    # Create RBFE Protocol class for measuring FE difference.
    rbfe_transform = RelativeHybridTopologyProtocol(settings=rbfe_settings)

    # Setup each edge of the network.
    runnable_edges = []

    for edge in network.edges:

        # Create four end points per edge.
        A_complex = ChemicalSystem({
            LIGAND: edge.componentA,
            SOLVENT: solvent,
            PROTEIN: protein,
        })
        A_solvent = ChemicalSystem({
            LIGAND: edge.componentA,
            SOLVENT: solvent,
        })

        B_complex = ChemicalSystem({
            LIGAND: edge.componentB,
            SOLVENT: solvent,
            PROTEIN: protein,
        })
        B_solvent = ChemicalSystem({
            LIGAND: edge.componentB,
            SOLVENT: solvent,
        })

        # Create two alchemical transformations per edge.
        complex_dag = rbfe_transform.create(
            stateA=A_complex,
            stateB=B_complex,
            mapping={LIGAND: edge},
        )
        solvent_dag = rbfe_transform.create(
            stateA=A_solvent,
            stateB=B_solvent,
            mapping={LIGAND: edge},
        )

        runnable_edges.append(RunnableEdge(edge.componentA, edge.componentB, solvent_dag, complex_dag))

    return runnable_edges, rbfe_transform


def _is_dry_run_result_success(dry_run_result: dict) -> bool:
    """Checks if the dry run is a success.

    Parameters
    ----------
    dry_run_result : dict
        The result of the dry run.

    Returns
    -------
    is_valid : bool
        True if the dry run result is valid.
        False if the dry run result is invalid.
    """
    is_valid = False

    if dry_run_result is not None and isinstance(
            dry_run_result, dict) and DEBUG in dry_run_result and dry_run_result[DEBUG][SAMPLER] is not None:
        is_valid = True

    return is_valid


def dry_run_edge(runnable_edge: RunnableEdge) -> EdgeSimulationResult:
    """Dry runs a single edge of the network.

    This involves setting can be useful when we want to check if the Simulation conditions would lead to a successful
    simulation, without running the simulation itself.

    Parameters
    ----------
    runnable_edge : RunnableEdge
        A single edge of the network to be run. A runnable_edge contains the
        components, solvent_dag, and complex_dag.

    Returns
    -------
    edgeSimulationResult : Type[EdgeSimulationResult]
        A dataclass containing the results of the simulation performed on the given edge.
    """
    componentA = runnable_edge.componentA
    componentB = runnable_edge.componentB
    solvent_dag = runnable_edge.solvent_dag
    complex_dag = runnable_edge.complex_dag

    complex_unit = list(complex_dag.protocol_units)[0]
    solvent_unit = list(solvent_dag.protocol_units)[0]

    dry_run_status = ''

    complex_dry_run_result = complex_unit.run(dry=True, verbose=True)
    solvent_dry_run_result = solvent_unit.run(dry=True, verbose=True)

    if _is_dry_run_result_success(solvent_dry_run_result) and _is_dry_run_result_success(complex_dry_run_result):
        dry_run_status = SUCCESS
    else:
        dry_run_status = FAILURE

    result = EdgeSimulationResult(
        componentA_name=componentA.name,
        componentB_name=componentB.name,
        is_dry_run=True,
        dry_run_status=dry_run_status,
    )

    return result


def run_edge(runnable_edge: RunnableEdge, rbfe_transform: Type[RelativeHybridTopologyProtocol]) -> EdgeSimulationResult:
    """Runs a single edge of the network.

    Parameters
    ----------
    runnable_edge : Type[RunnableEdge]
        A single edge of the network to be run. A runnable_edge contains the
        components, solvent_dag, and complex_dag.
    rbfe_transform : Type[RelativeHybridTopologyProtocol]
        A RelativeHybridTopologyProtocol instance used to gather the results generated from executing the solvent and the complex Dags.


    Returns
    -------
    edgeSimulationResult : Type[EdgeSimulationResult]
        A dataclass containing the results of the simulation performed on the given edge.
    """

    componentA = runnable_edge.componentA
    componentB = runnable_edge.componentB
    solvent_dag = runnable_edge.solvent_dag
    complex_dag = runnable_edge.complex_dag

    # Create new directories for outputs.
    solvent_path = Path('./solvent')
    complex_path = Path('./complex')
    solvent_path.mkdir(exist_ok=True)
    complex_path.mkdir(exist_ok=True)

    # Execute the solvent transformation
    solvent_dag_results = execute_DAG(solvent_dag,
                                      scratch_basedir=solvent_path,
                                      shared_basedir=solvent_path,
                                      keep_scratch=True)

    # Execute the complex transformation
    complex_dag_results = execute_DAG(complex_dag,
                                      scratch_basedir=complex_path,
                                      shared_basedir=complex_path,
                                      keep_scratch=True)

    # Get the complex and solvent results
    complex_results = rbfe_transform.gather([complex_dag_results])
    solvent_results = rbfe_transform.gather([solvent_dag_results])

    complex_dG = complex_results.get_estimate()
    solvent_dG = solvent_results.get_estimate()
    edge_ddG = complex_dG - solvent_dG

    complex_dG_uncertainty = complex_results.get_uncertainty()
    solvent_dG_uncertainty = solvent_results.get_uncertainty()
    edge_ddG_uncertainty = (complex_dG_uncertainty**2 + solvent_dG_uncertainty**2)**0.5

    edgeSimulationResult = EdgeSimulationResult(componentA_name=componentA.name,
                                                componentB_name=componentB.name,
                                                is_dry_run=False,
                                                complex_dG=complex_dG,
                                                complex_dG_uncertainty=complex_dG_uncertainty,
                                                solvent_dG=solvent_dG,
                                                solvent_dG_uncertainty=solvent_dG_uncertainty,
                                                edge_ddG=edge_ddG,
                                                edge_ddG_uncertainty=edge_ddG_uncertainty)

    return edgeSimulationResult
