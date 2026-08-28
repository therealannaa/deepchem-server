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

import json
import logging
import os
import tempfile
from typing import Dict, List, Union

from openff.units import unit  # type: ignore
import pandas as pd  # type: ignore
import pint

from deepchem_server.core.common import config
from deepchem_server.core.common.cards import DataCard
from deepchem_server.core.primitives.fep.rbfe.data_domain_classes.ExperimentalMeasurement import ExperimentalMeasurement


logger = logging.getLogger(__name__)


def collate_rbfe_results(result_file_deepchem_addresses: List[str], reference_ligand: Union[None, str],
                         reference_ligand_dg_value: Union[str, None],
                         reference_ligand_dg_value_uncertainty: Union[str, None], output_file_name: Union[None, str]):
    """Primitive function to collate RBFE results.

    This function takes in a list of RBFE result files' addresses, processes them, and creates a CSV
    containing the list of all Ligands, their DG Values, and the Calculation Uncertainty(Error) as deduced
    from the result files.

    The function ignores any non-relevant files passed in as input. Invalid files do not affect the execution flow.

    The reference_ligand field is Optional. However, if the reference_ligand field is specified, the other
    reference_ligand_dg_value and reference_ligand_dg_value_uncertainty fields MUST be specified.

    Parameters
    ----------
    result_file_deepchem_addresses : List[str]
        The list of RBFE result files to be analyzed
    reference_ligand : Union[None, str]
        The name of the reference ligand whose experimental dG value is known
    reference_ligand_dg_value : str
        The dG value of the reference ligand in a pint.Quantity format (Example '-9.00 kcal/mol')
    reference_ligand_dg_value_uncertainty : str
        The uncertainty(error) in the dG value of the reference ligand in a pint.Quantity format (Example '0.01 kcal/mol')
    output_file_name : Union[None, str]
        Name of the output result file

    Returns
    -------
    result_file_address
        THe address of the saved result file.

    Raises
    ------
    ValueError
        If no valid RBFE Results file address was specified.
    """

    # Handle when the string "None" is passed as the arguments, This will happen since we use Program String to submit jobs, and None is not directly supported by our Parser.
    if isinstance(reference_ligand, str) and reference_ligand.strip() == 'None':
        reference_ligand = None
    if isinstance(reference_ligand_dg_value, str) and reference_ligand_dg_value.strip() == 'None':
        reference_ligand_dg_value = None
    if isinstance(reference_ligand_dg_value_uncertainty,
                  str) and reference_ligand_dg_value_uncertainty.strip() == 'None':
        reference_ligand_dg_value_uncertainty = None

    # Handle a None or Empty reference_ligand
    if not reference_ligand:
        reference_ligand = None
        reference_ligand_dg_value = '0 kcal/mol'
        reference_ligand_dg_value_uncertainty = '0 kcal/mol'
    else:
        assert reference_ligand_dg_value is not None and reference_ligand_dg_value is not None, "If the reference_ligand parameter is specified, reference_ligand_dg_value and reference_ligand_dg_value_uncertainty parameters MUST be specified."

    # Handle a None or Empty ligand_dg_value
    if not reference_ligand_dg_value:
        reference_ligand_dg_value = '0 kcal/mol'

    # Handle a None or Empty reference_ligand_dg_value_uncertainty value
    if not reference_ligand_dg_value_uncertainty:
        reference_ligand_dg_value_uncertainty = '0 kcal/mol'

    assert result_file_deepchem_addresses, "result_file_deepchem_addresses cannot be empty or None"

    assert output_file_name, "output_file_name cannot be empty or None"
    if not output_file_name.endswith(".csv"):
        output_file_name += ".csv"

    # Get the list of simulation results after processing all the input files
    simulation_results: List = process_input_files(result_file_deepchem_addresses)

    if len(simulation_results) <= 0:
        raise ValueError("No valid Simulation Result was found.")

    # Calculate the dG values for each ligand from the simulation results
    ligand_dg_values = calculate_dg_values(
        simulation_results,
        reference_ligand,  # type: ignore
        reference_ligand_dg_value,
        reference_ligand_dg_value_uncertainty)

    result_dataframe = get_result_dataframe(ligand_dg_values)

    # Save the results as a csv file
    result_dataframe.to_csv('temp.csv', index=False)

    result_datacard = DataCard(
        address='',
        file_type='csv',
        data_type='pandas.DataFrame',
    )

    # Upload the CSV file to the user's deepchem datastore
    saved_csv_file_name = config.get_datastore().upload_data(  # type: ignore
        output_file_name, 'temp.csv', result_datacard)

    # Delete the temporary file
    os.remove('temp.csv')

    return saved_csv_file_name


def process_input_files(result_file_deepchem_addresses: list[str]):
    """Helper function for collate_rbfe_results()

    This function takes in a list of RBFE Result file addresses, fetched them from the deepchem datastore.
    It then parses each result file, ignoring the invalid file, and creates an array containing the parsed
    result objects.
    The list of results is returned.

    Parameters
    ----------
    result_file_deepchem_addresses : list[str]
        The list of deepchem addresses of all result files to be processed.

    Returns
    -------
    simulation_results
        A list of dictionaries, each dictionary corresponding to a EdgeSimulationResult Object

    Raises
    ------
    Exception
        Raises a Datastore not set exception if the deepchem datastore is not set.
    """

    # Raise exception if the datastore is not set
    if config.get_datastore() is None:
        raise Exception("Datastore not set")

    fetched_result_files = []
    simulation_results = []

    tempdir = tempfile.TemporaryDirectory()

    for deepchem_file_path in result_file_deepchem_addresses:
        try:
            file_name = os.path.basename(deepchem_file_path)
            temp_file_path = os.path.join(tempdir.name, file_name)
            config.get_datastore().download_object(  # type: ignore
                file_name, temp_file_path)
            fetched_result_files.append(temp_file_path)
        except Exception:
            logger.error(f"Could not fetch file - {deepchem_file_path}")

    for result_file_name in fetched_result_files:
        try:
            with open(result_file_name) as result_file:
                result_data = json.load(result_file)
                assert result_data['componentA_name'] is not None
                assert result_data['componentB_name'] is not None
                assert result_data['edge_ddG'] is not None
                assert result_data['edge_ddG_uncertainty'] is not None

                simulation_results.append(result_data)
        except Exception:
            logger.warning(
                f"Could not process {result_file_name}. Please ensure that the specified file is a valid RBFE result file. Dry Run result files are ignored."
            )

    return simulation_results


def get_ligands_from_results(simulation_results: list[dict]):
    """Helper function for collate_rbfe_results(). Extracts the ligand names from the simulation results.

    This function extracts the ligand names from the simulation results and return a list of these names.

    Parameters
    ----------
    simulation_results : list[dict]
        The list of simulation_results obtained from process_input_files()

    Returns
    -------
    ligands
        A list containing a non-duplicate list of all the ligands extracted from the simulation_results.
    """

    ligands = []

    for simulation_result in simulation_results:
        ligands.append(simulation_result['componentA_name'])
        ligands.append(simulation_result['componentB_name'])

    # Remove duplicates, and return as a list
    return list(set(ligands))


def calculate_dg_values(simulation_results: list[dict], reference_ligand: str, reference_ligand_dg_value: str,
                        reference_ligand_dg_value_uncertainty: str):
    """Helper function for collate_rbfe_results(). Calculates the dG values for each ligand from the simulation_results.

    This function takes in the simulation_results object generated from the process_input_files() function, along with
    reference_ligand, reference_ligand_dg_value and reference_ligand_dg_value_uncertainty.

    It then calculates the the dG value for each possible ligand in the simulation results.
    The following equation forms the base for each calculation

    DDG(Component_A -> Component_B) = dG(Component_B) - dG(Component_A)

    If a valid reference ligand is not specified, The first ligand in the list of ligands is set as the
    reference ligand with a dG value of 0.0 KCal/Mol.

    Parameters
    ----------
    simulation_results : list[dict]
        The list of simulation_results obtained from process_input_files()
    reference_ligand : Union[None, str]
        The name of the reference ligand whose experimental dG value is known
    reference_ligand_dg_value : str
        The dG value of the reference ligand in a pint.Quantity format (Example '-9.00 kcal/mol')
    reference_ligand_dg_value_uncertainty : str
        The uncertainty(error) in the dG value of the reference ligand in a pint.Quantity format (Example '0.01 kcal/mol')

    Returns
    -------
    ligand_dg_values
        A dictionary containing the ligand -> ExperimentalMeasurement(Value, Uncertainty) mapping for each ligand
    """

    ligand_dg_values: Dict[str, ExperimentalMeasurement] = {}

    ligands = get_ligands_from_results(simulation_results)

    if reference_ligand is None or reference_ligand not in ligands:
        logger.warning(
            f"Either reference ligand is None, or specified Reference Ligand not found in the set of results, and will be IGNORED. Setting {ligands[0]} as the Reference Ligand with a DG Value of 0 Kcal/Mol"
        )
        reference_ligand = ligands[0]
        reference_ligand_dg_value = "0.00 kcal/mol"
        reference_ligand_dg_value_uncertainty = "0.00 kcal/mol"

    for ligand in ligands:
        ligand_dg_values[ligand] = None  # type: ignore

    ligand_dg_values[reference_ligand] = ExperimentalMeasurement(reference_ligand_dg_value,
                                                                 reference_ligand_dg_value_uncertainty)

    # Maintain a while loop iteration count to prevent an infinite loop
    iteration_count = 0

    while True:
        ligand_dg_values_copy = dict(ligand_dg_values)

        for simulation_result in simulation_results:
            if ligand_dg_values[simulation_result['componentA_name']] is not None and ligand_dg_values[
                    simulation_result['componentB_name']] is None:
                ligand_dg_values[simulation_result['componentB_name']] = ExperimentalMeasurement(
                    value=pint.Quantity(simulation_result['edge_ddG']) +  # type: ignore
                    ligand_dg_values[simulation_result['componentA_name']].value,
                    uncertainty=pint.Quantity(  # type: ignore
                        simulation_result['edge_ddG_uncertainty']) +
                    ligand_dg_values[simulation_result['componentA_name']].uncertainty)

            if ligand_dg_values[simulation_result['componentA_name']] is None and ligand_dg_values[
                    simulation_result['componentB_name']] is not None:
                ligand_dg_values[simulation_result[
                    'componentA_name']] = ExperimentalMeasurement(
                        value=ligand_dg_values[
                            simulation_result['componentB_name']].value -
                        pint.Quantity(simulation_result['edge_ddG']
                                     ),  # type: ignore
                        uncertainty=ligand_dg_values[
                            simulation_result['componentB_name']].uncertainty +
                        pint.Quantity(  # type: ignore
                            simulation_result['edge_ddG_uncertainty'])
                )  # yapf: disable

        iteration_count += 1

        # If the ligand_dg_values object has not changed, break the loop
        if ligand_dg_values == ligand_dg_values_copy:
            break

        # If the iteration count exceeds Ten times the length of the number of edges, break the loop.
        if iteration_count >= 10 * len(simulation_results):
            logger.error("An unexpected error occurred while calculating the dG values. Exiting!")
            break

    return ligand_dg_values


def get_result_dataframe(ligand_dg_values: dict):
    """Helper function for the collate_rbfe_results() function. Creates a dataframe out of the ligand_dg_values object.

    This function takes in the ligand_dg_values dict generated by the calculate_dg_values() function, and then
    created a pandas datafrane the Ligand Name, DG Value and the calculation Uncertainty, sorted by the descending order
    of dG Values.

    Parameters
    ----------
    ligand_dg_values : dict
       A dictionary containing the ligand -> ExperimentalMeasurement(Value, Uncertainty) mapping for each ligand
       Generated by the calculate_dg_values() function

    Returns
    -------
    result_dataframe
        A pandas dataframe containing the calculated dG values and uncertainties for all ligands, ordered by their dG values.
    """
    result_dataframe = pd.DataFrame(columns=['Ligand', 'DG Value', 'Uncertainty'])

    for key, value in ligand_dg_values.items():

        if value is None:
            logger.warning(f"Information insufficient to calculate DG value for {key}")
            dg_value = None
            uncertainty = None
        else:
            dg_value = pint.Quantity(value.value).to(  # type: ignore
                unit.kilocalorie / unit.mole)
            uncertainty = pint.Quantity(value.uncertainty).to(  # type: ignore
                unit.kilocalorie / unit.mole)
        result_dataframe.loc[len(result_dataframe)] = {
            'Ligand': key,
            'DG Value': dg_value and dg_value.round(4),  # type: ignore
            'Uncertainty':
                uncertainty and uncertainty.round(4)  # type: ignore
        }

    result_dataframe.sort_values(by='DG Value', ascending=False, inplace=True)
    result_dataframe.reset_index(drop=True, inplace=True)

    return result_dataframe
