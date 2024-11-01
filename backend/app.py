import logging
from apscheduler.schedulers.background import BackgroundScheduler

from my_logging import configure_root_logger

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)
configure_root_logger()

from functools import partial

import gradio as gr
# Choose ligtheval or harness backend
# from main_backend_lighteval import run_auto_eval
from main_backend_harness import run_auto_eval

from src.display.log_visualizer import log_file_to_html_string
from src.display.css_html_js import dark_mode_gradio_js
from src.envs import REFRESH_RATE, REPO_ID, QUEUE_REPO, RESULTS_REPO
from my_logging import setup_logger, log_file

logging.basicConfig(level=logging.INFO)
logger = setup_logger(__name__)


intro_md = f"""
# Intro
This is a visual for the auto evaluator. 
"""

links_md = f"""
# Important links

| Description     | Link |
|-----------------|------|
| Leaderboard     | [{REPO_ID}](https://huggingface.co/spaces/{REPO_ID}) |
| Queue Repo      | [{QUEUE_REPO}](https://huggingface.co/datasets/{QUEUE_REPO}) |
| Results Repo    | [{RESULTS_REPO}](https://huggingface.co/datasets/{RESULTS_REPO}) |
"""

def auto_eval():
    logger.info("Triggering Auto Eval")
    run_auto_eval()


reverse_order_checkbox = gr.Checkbox(label="Reverse Order", value=True)

with gr.Blocks(js=dark_mode_gradio_js) as demo:
    gr.Markdown(intro_md)
    with gr.Tab("Application"):
        output_html = gr.HTML(partial(log_file_to_html_string, reverse=reverse_order_checkbox), every=1)
        with gr.Row():
            download_button = gr.DownloadButton("Download Log File", value=log_file)
            with gr.Accordion('Log View Configuration', open=False):
                reverse_order_checkbox.render()
        # Add a button that when pressed, triggers run_auto_eval
        button = gr.Button("Manually Run Evaluation")
        gr.Markdown(links_md)

        #dummy = gr.Markdown(auto_eval, every=REFRESH_RATE, visible=False)

        button.click(fn=auto_eval, inputs=[], outputs=[])

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(auto_eval, "interval", seconds=REFRESH_RATE)
    scheduler.start()
    demo.queue(default_concurrency_limit=40).launch(server_name="127.0.0.1",
                                                          show_error=True,
                                                          server_port=9870)