import os

from huggingface_hub import HfApi

# Info to change for your repository
# ----------------------------------
TOKEN = os.environ.get("HF_TOKEN") # A read/write token for your org

OWNER = "zhouzl" # Change to your org - don't forget to create a results and request dataset

# For harness evaluations
DEVICE = "cuda:0" # "cuda:0" if you add compute, for harness evaluations
LIMIT = 20 # !!!! For testing, should be None for actual evaluations!!!
NUM_FEWSHOT = 0 # Change with your few shot for the Harness evaluations
TASKS_HARNESS = ["anli_r1", "logiqa"]

# For lighteval evaluations
ACCELERATOR = "cpu"
REGION = "us-east-1"
VENDOR = "aws"
TASKS_LIGHTEVAL = "lighteval|anli:r1|0|0,lighteval|logiqa|0|0" 
# To add your own tasks, edit the custom file and launch it with `custom|myothertask|0|0``

# ---------------------------------------------------
REPO_ID = f"{OWNER}/backend"
QUEUE_REPO = f"{OWNER}/requests"
RESULTS_REPO = f"{OWNER}/results"

# If you setup a cache later, just change HF_HOME
CACHE_PATH=os.getenv("HF_HOME", ".")

# Local caches
EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval-results")
EVAL_REQUESTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-queue-bk")
EVAL_RESULTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-results-bk")

REFRESH_RATE = 10 * 60  # 10 min
NUM_LINES_VISUALIZE = 300

API = HfApi(token=TOKEN)

