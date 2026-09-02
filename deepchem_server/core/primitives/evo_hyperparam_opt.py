"""Evolutionary Hyperparameter Search Inspired by Co-Scientist"""
import ast
import json
import random
import copy
import uuid
import math
import numpy as np
from typing import Dict, List, Optional, Any

import deepchem as dc
from deepchem_server.core.common import config, model_mappings
from deepchem_server.core.common.cards import ModelCard, DataCard
from deepchem_server.core.common.address import DeepchemAddress
from deepchem_server.core.primitives.evaluator import deepchem_server_metrics

MIN_METRIC_LIST = ["rms_score", "mae_error"]

def _sample_param(space_def: Any) -> Any:
    """Sample a parameter based on space definition (categorical list or dict w/ bounds)."""
    if isinstance(space_def, list):
        return random.choice(space_def)
    elif isinstance(space_def, dict) and 'low' in space_def and 'high' in space_def:
        low = space_def['low']
        high = space_def['high']
        is_int = space_def.get('type') == 'int'
        is_log = space_def.get('log', False)
        
        if is_log:
            val = math.exp(random.uniform(math.log(low), math.log(high)))
        else:
            val = random.uniform(low, high)
            
        if is_int:
            return int(round(val))
        return val
    else:
        return space_def

def _mutate_params(params: Dict[str, Any], space: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
    """Mutate parameters using uniform or gaussian stepping logic."""
    mutated = copy.deepcopy(params)
    for key, space_def in space.items():
        if random.random() < mutation_rate:
            if isinstance(space_def, list):
                if len(space_def) > 1:
                    choices = [v for v in space_def if v != mutated[key]]
                    if choices:
                        mutated[key] = random.choice(choices)
            elif isinstance(space_def, dict) and 'low' in space_def and 'high' in space_def:
                low = space_def['low']
                high = space_def['high']
                is_int = space_def.get('type') == 'int'
                is_log = space_def.get('log', False)
                # Small gaussian step (10% of bounded range)
                if is_log:
                    current_log = math.log(mutated[key])
                    step = (math.log(high) - math.log(low)) * random.gauss(0, 0.1)
                    new_val = math.exp(current_log + step)
                else:
                    step = (high - low) * random.gauss(0, 0.1)
                    new_val = mutated[key] + step
                # Clip bounds
                new_val = max(low, min(high, new_val))
                if is_int:
                    mutated[key] = int(round(new_val))
                else:
                    mutated[key] = new_val
    return mutated

def evo_hyperparam_opt(model_type: str,
                       train_address: str,
                       valid_address: str,
                       hyperparams_space: Dict,
                       output_prefix: str,
                       metric: str = 'pearson_r2_score',
                       population_size: int = 4,
                       generations: int = 3,
                       nb_epoch: int = 10,
                       max_evals: int = 100,
                       seed: Optional[int] = None):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if isinstance(hyperparams_space, str):
        hyperparams_space = ast.literal_eval(hyperparams_space)
    
    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError("Datastore not set")
    
    train_dataset = datastore.get(train_address)
    valid_dataset = datastore.get(valid_address)

    if model_type not in model_mappings.model_address_map:
        raise ValueError("Model type not recognized.")

    def _model_builder(**model_params):
        return model_mappings.model_address_map[model_type](**model_params)

    use_max = metric not in MIN_METRIC_LIST
    metric_obj = deepchem_server_metrics[metric]

    population = []
    # Generation 0 initialization
    for _ in range(population_size):
        params = {k: _sample_param(v) for k, v in hyperparams_space.items()}
        population.append({
            'config_id': str(uuid.uuid4()),
            'parent_id': None,
            'params': params
        })

    all_results = []
    best_overall_score = float('-inf') if use_max else float('inf')
    best_overall_params = None
    best_overall_config_id = None
    total_evals = 0

    for gen in range(generations):
        gen_results = []
        for individual in population:
            if total_evals >= max_evals:
                continue
                
            params = individual['params']
            config_id = individual['config_id']
            parent_id = individual['parent_id']
            
            eval_epochs = max(1, nb_epoch // 2)
            model = _model_builder(**params)
            
            try:
                # Train model
                if 'nb_epoch' in model.__class__.fit.__code__.co_varnames:
                    model.fit(train_dataset, nb_epoch=eval_epochs)
                else:
                    model.fit(train_dataset)
                score = model.evaluate(valid_dataset, [metric_obj])[metric_obj.name]
            except Exception as e:
                score = float('-inf') if use_max else float('inf')
                
            total_evals += 1
            
            record = {
                'generation': gen,
                'config_id': config_id,
                'parent_id': parent_id,
                'params': params,
                'score': score
            }
            gen_results.append(record)
            all_results.append(record)

            is_better = (score > best_overall_score) if use_max else (score < best_overall_score)
            if is_better:
                best_overall_score = score
                best_overall_params = copy.deepcopy(params)
                best_overall_config_id = config_id
                
        if total_evals >= max_evals:
            break

        # Sort population by score (fitness)
        gen_results.sort(key=lambda x: x['score'], reverse=use_max)
        
        # Setup Next Generation if we have more bounds
        if gen < generations - 1 and gen_results:
            next_population = []
            
            # 1. Elitism Protocol
            elite = gen_results[0]
            next_population.append({
                'config_id': str(uuid.uuid4()),
                'parent_id': elite['config_id'],
                'params': copy.deepcopy(elite['params'])
            })
            
            # 2. Exploration (Random immigrant injection mechanism)
            num_immigrants = max(1, int(0.2 * population_size))
            
            # 3. Exploitation (Tournament Selection / Crossover Mutations)
            top_k = max(1, population_size // 2)
            survivors = gen_results[:top_k]
            
            while len(next_population) < population_size - num_immigrants:
                parent = random.choice(survivors)
                child_params = _mutate_params(parent['params'], hyperparams_space, mutation_rate=0.4)
                next_population.append({
                    'config_id': str(uuid.uuid4()),
                    'parent_id': parent['config_id'],
                    'params': child_params
                })
                
            # Create Immigrants
            while len(next_population) < population_size:
                immigrant_params = {k: _sample_param(v) for k, v in hyperparams_space.items()}
                next_population.append({
                    'config_id': str(uuid.uuid4()),
                    'parent_id': 'immigrant',
                    'params': immigrant_params
                })
                
            population = next_population

    if best_overall_params is None:
        raise RuntimeError("Evolutionary search failed to find any valid configurations.")

    # Final Training run on best overall parameters with the full epoch budget
    best_model = _model_builder(**best_overall_params)
    if 'nb_epoch' in best_model.__class__.fit.__code__.co_varnames:
        best_model.fit(train_dataset, nb_epoch=nb_epoch)
    else:
        best_model.fit(train_dataset)
        
    best_model.save()

    # Logging Best Model
    model_card = ModelCard(address='',
                           model_type=model_type,
                           train_dataset_address=train_address,
                           valid_dataset_address=valid_address,
                           init_kwargs=best_overall_params,
                           train_kwargs={})
    model_name = DeepchemAddress.get_key(output_prefix) + '_best_model'
    model_address = datastore.upload_data_from_memory(best_model, model_name, model_card, kind='model')

    # Export configuration and full lineage trajectory 
    best_hyperparams_str = json.dumps({
        'config_id': best_overall_config_id,
        'params': best_overall_params,
        'score': best_overall_score
    })
    card_hp = DataCard(address='', file_type='json', data_type='json', description="best hyperparams from evo search")
    output_address_best_hyperparams = datastore.upload_data_from_memory(
        best_hyperparams_str,
        DeepchemAddress.get_key(output_prefix) + '_best_hyperparams.json', card_hp)

    all_results_str = json.dumps(all_results)
    card_res = DataCard(address='', file_type='json', data_type='json', description="lineage tree of evo search")
    output_address = datastore.upload_data_from_memory(
        all_results_str,
        DeepchemAddress.get_key(output_prefix) + '_all_results.json',
        card_res)
    
    return model_address, output_address_best_hyperparams, output_address
