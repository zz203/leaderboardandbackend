import json
import argparse
import logging
from datetime import datetime

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_config import InferenceEndpointModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

from lighteval.main_accelerate import main, EnvConfig, create_model_config

from src.envs import RESULTS_REPO, CACHE_PATH, TOKEN
from src.backend.manage_requests import EvalRequest
from my_logging import setup_logger

logging.getLogger("openai").setLevel(logging.WARNING)
logger = setup_logger(__name__)

def run_evaluation(eval_request: EvalRequest, task_names: str, batch_size: int, local_dir: str, accelerator: str, region: str, vendor: str, instance_size: str, instance_type: str, limit=None):
    """Runs one evaluation for the current evaluation request file using lighteval, then pushes the results to the hub.

    Args:
        eval_request (EvalRequest): Input evaluation request file representation
        task_names (list): Tasks to launch
        batch_size (int): Selected batch size
        accelerator (str): Inference endpoint parameter for running the evaluation
        region (str):  Inference endpoint parameter for running the evaluation
        vendor (str):  Inference endpoint parameter for running the evaluation
        instance_size (str):  Inference endpoint parameter for running the evaluation
        instance_type (str):  Inference endpoint parameter for running the evaluation
        local_dir (str): Where to save the results locally
        no_cache (bool, optional): Whether to use a cache or not.
        limit (int, optional): Whether to use a number of samples only for the evaluation - only for debugging
    """    

    if limit:
        logger.info("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details = True,
        push_to_hub = True,
        push_to_tensorboard = False,
        hub_results_org= RESULTS_REPO,
        public = False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        override_batch_size=batch_size,
        max_samples=limit,
        use_chat_template=False,
        system_prompt=None,
        custom_tasks_directory="custom_tasks.py", # if using a custom task
    )

    model_config = InferenceEndpointModelConfig(
        # Endpoint parameters
        name = eval_request.model.replace(".", "-").lower(), 
        repository = eval_request.model,
        accelerator =  accelerator,
        vendor= vendor,
        region= region,
        instance_size= instance_size,
        instance_type= instance_type,
        should_reuse_existing= False,
        model_dtype= eval_request.precision,
        revision= eval_request.revision,
    )

    pipeline = Pipeline(
        tasks=task_names,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    try:
        pipeline.evaluate()
        pipeline.show_results()
        pipeline.save_and_push_results()
        results = pipeline.get_results()

        dumped = json.dumps(results, indent=2)
        logger.info(dumped)

    except Exception as e: # if eval failed, we force a cleanup
        pipeline.model.cleanup()

    return results
