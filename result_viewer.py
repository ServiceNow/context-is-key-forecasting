import argparse
import gradio as gr

from pathlib import Path


RESULTS_PATH = None


def load_methods():
    """
    Load all methods available in the results directory

    """
    methods = sorted([p.name for p in Path(RESULTS_PATH).iterdir() if p.is_dir()])
    return gr.Dropdown(choices=methods, value=methods[0])


def load_tasks(method):
    """
    Updates the task dropdown with all tasks available for a given method

    Parameters:
    -----------
    method: str
        Name of the method to load tasks for

    """
    method_path = Path(RESULTS_PATH) / method
    tasks = sorted([p.name for p in method_path.iterdir() if p.is_dir()])
    return gr.Dropdown(choices=tasks, value=tasks[0])


def load_task_seeds(method, task):
    """
    Updates the seed dropdown with all seeds available for a given method and task

    Parameters:
    -----------
    method: str
        Name of the method to load tasks for
    task: str
        Name of the task to load seeds for

    """
    task_path = Path(RESULTS_PATH) / method / task
    seeds = sorted(list(set([p.stem for p in task_path.iterdir() if p.is_dir()])))
    return gr.Dropdown(choices=seeds, value=seeds[0])


def load_result(method, task, seed):
    """
    Load the forecast plot and context for a given method, task, and seed

    Parameters:
    -----------
    method: str
        Name of the method to load tasks for
    task: str
        Name of the task to load seeds for
    seed: str
        Name of the seed to load results

    """
    seed_path = Path(RESULTS_PATH) / method / task / seed
    forecast_plot_file = seed_path / "forecast.png"

    with open(seed_path / "context", "r") as f:
        context = f.read()

    return forecast_plot_file, context


def main():
    with gr.Blocks(theme=gr.themes.Soft()) as main_ui:
        # Result selection dropdowns
        methods_dropdown = gr.Dropdown(label="Method", interactive=True)
        tasks_dropdown = gr.Dropdown(label="Task", interactive=True)
        seed_dropdown = gr.Dropdown(label="Seed", interactive=True)

        with gr.Row():
            forecast_image = gr.Image(label="Forecast", height=500, width=500)
            context_text = gr.TextArea(label="Context")

        # Main interface events
        main_ui.load(fn=load_methods, outputs=[methods_dropdown])

        # Dropdown dynamic loading events
        methods_dropdown.change(
            fn=load_tasks, inputs=[methods_dropdown], outputs=tasks_dropdown
        )
        tasks_dropdown.change(
            fn=load_task_seeds,
            inputs=[methods_dropdown, tasks_dropdown],
            outputs=[seed_dropdown],
        )
        seed_dropdown.change(
            fn=load_result,
            inputs=[methods_dropdown, tasks_dropdown, seed_dropdown],
            outputs=[forecast_image, context_text],
        )

    main_ui.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-path", type=str, required=True, help="Path to the results directory"
    )
    args = parser.parse_args()

    RESULTS_PATH = args.results_path

    main()
