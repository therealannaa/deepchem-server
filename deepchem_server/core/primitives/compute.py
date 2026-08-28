from typing import Any, Dict

from deepchem_server.core.primitives.del_denoising import del_denoise
from deepchem_server.core.primitives.docking import generate_pose
from deepchem_server.core.primitives.evaluator import model_evaluator
from deepchem_server.core.primitives.feat import featurize
from deepchem_server.core.primitives.inference import infer
from deepchem_server.core.primitives.partition import partition
from deepchem_server.core.primitives.splitter import train_valid_test_split
from deepchem_server.core.primitives.train import train
from deepchem_server.core.primitives.transform import transform


def lazy_run_rbfe(*args, **kwargs):
    from deepchem_server.core.primitives.fep.rbfe.run_rbfe import run_rbfe as _run_rbfe
    return _run_rbfe(*args, **kwargs)


def lazy_collate_rbfe_results(*args, **kwargs):
    from deepchem_server.core.primitives.fep.rbfe.collate_rbfe_results import collate_rbfe_results as _collate_rbfe_results
    return _collate_rbfe_results(*args, **kwargs)


program_map = {
    "featurize": featurize,
    "train": train,
    "evaluate": model_evaluator,
    "infer": infer,
    "partition": partition,
    "train_valid_test_split": train_valid_test_split,
    "generate_pose": generate_pose,
    "relative_binding_free_energy": lazy_run_rbfe,
    "collate_rbfe_results": lazy_collate_rbfe_results,
    "del_denoise": del_denoise,
    "transform": transform,
}


class ComputeWorkflow:
    """A Compute Workflow is a workflow that runs on Deepchem Server.

    Parameters
    ----------
    program : Dict
        A dictionary containing program configuration including 'program_name'
        and other parameters required by the specific program.

    Examples
    --------
    >>> program = {
    ...     'program_name': 'featurize',
    ...     'dataset_address': 'deepchem://profile_name/project_name/data.csv',
    ...     'featurizer': 'ecfp',
    ...     'output': 'test_output',
    ...     'dataset_column': 'smiles',
    ...     'feat_kwargs': {'size': 1024},
    ...     'label_column': 'y',
    ... }
    >>> workflow = ComputeWorkflow(program)
    """

    def __init__(self, program: Dict) -> None:
        """Initialize ComputeWorkflow.

        Parameters
        ----------
        program : Dict
            A dictionary containing program configuration.
        """
        self.program: Dict = program

    def execute(self) -> Any:
        """Run the program based on the 'program_name' in the program dictionary.

        Returns
        -------
        Any
            The output of the executed program.

        Raises
        ------
        ValueError
            If 'program_name' is not found in program or if the program_name
            is not available in the program map.
        """
        if 'program_name' not in self.program:
            raise ValueError("program_name not found in program")
        program_name: str = self.program['program_name']
        params: Dict = {key: value for key, value in self.program.items() if key != 'program_name'}
        if program_name not in program_map:
            raise ValueError(f"{program_name} not in available programs")

        output: Any = program_map[program_name](**params)  # type: ignore
        return output
