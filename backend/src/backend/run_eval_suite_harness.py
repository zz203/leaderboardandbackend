import json
import os
import logging
from datetime import datetime

from lm_eval import tasks, evaluator, utils
from lm_eval.tasks import TaskManager

from src.envs import RESULTS_REPO, API
from src.backend.manage_requests import EvalRequest
from my_logging import setup_logger

from typing import Union

logging.getLogger("openai").setLevel(logging.WARNING)
logger = setup_logger(__name__)

# def make_table(results):
#     import pandas as pd
#     # 假设 results 是一个字典，包含任务名称和对应的分数
#     table = pd.DataFrame.from_dict(results, orient='index')
#     return table.to_string()

def run_evaluation(eval_request: EvalRequest, task_names: list, num_fewshot: int, batch_size: Union[int, str], device: str, local_dir: str, results_repo: str, no_cache: bool =True, limit: int =None):
    """Runs one evaluation for the current evaluation request file, then pushes the results to the hub.

    Args:
        eval_request (EvalRequest): Input evaluation request file representation
        task_names (list): Tasks to launch
        num_fewshot (int): Number of few shots to use
        batch_size (int or str): Selected batch size or 'auto'
        device (str): "cpu" or "cuda:0", depending on what you assigned to the space
        local_dir (str): Where to save the results locally
        results_repo (str): To which repository to upload the results
        no_cache (bool, optional): Whether to use a cache or not
        limit (int, optional): Whether to use a number of samples only for the evaluation - only for debugging

    Returns:
        _type_: _description_
    """
    if limit:
        logger.info(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    task_manager = TaskManager()
    all_tasks = task_manager.all_tasks
    task_names = utils.pattern_match(task_names, all_tasks)

    logger.info(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=eval_request.get_model_args(),
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        write_out=True # Whether to write out an example document and model input, for checking task integrity
    )

    results["config"]["model_dtype"] = eval_request.precision
    results["config"]["model_name"] = eval_request.model
    results["config"]["model_sha"] = eval_request.revision

    dumped = json.dumps(results, indent=2)
    logger.info('results dumped!')
    # logger.info(dumped)
    # logger.info(f"Results type: {type(results)}")
    # logger.info(f"Results content: {results}")

    # output_path = os.path.join(local_dir, *eval_request.model.split("/"), f"results_{datetime.now()}.json")
    timestamp = datetime.now().strftime("%Y-%m-%d %H_%M_%S.%f")
    output_path = os.path.join(local_dir, *eval_request.model.split("/"), f"results_{timestamp}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(dumped)

    # logger.info(make_table(results))
    print("-----------uploading----------")
    timestamp = datetime.now().strftime("%Y-%m-%d %H_%M_%S.%f")
    API.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=f"{eval_request.model}/results_{timestamp}.json",
        repo_id=results_repo,
        repo_type="dataset",
    )
    print('path_or_fileobj', output_path)
    print('path_in_repo', f"{eval_request.model}/results_{timestamp}.json")
    print('repo_id', results_repo)
    print("repo_type=dataset")
    print("-----------finish uploading----------")

    return results
