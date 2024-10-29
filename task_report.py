import argparse
import base64
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from benchmark import ALL_TASKS
from benchmark.base import ALLOWED_SKILLS
from datetime import datetime
from io import BytesIO


CONTEXT_PRETTY_NAMES = {
    "c_h": "History",
    "c_f": "Future",
    "c_i": "Intemporal",
    "c_cov": "Covariates",
    "c_causal": "Causal",
}


def _figure_to_html(figure):
    """
    Convert a matplotlib figure to an HTML string

    """
    # Save the figure to a byte array
    figure_bytes = BytesIO()
    figure.savefig(figure_bytes, format="png")
    plt.close(figure)
    figure_bytes.seek(0)

    # Convert byte array to base64 string
    figure_base64 = base64.b64encode(figure_bytes.read()).decode("utf-8")

    # Render as it would be in an HTML file
    figure_html = f'<img src="data:image/png;base64,{figure_base64}" class="img-fluid" alt="Figure"/>'

    return figure_html


def get_task_context_info(tasks):
    """
    Get a DataFrame with information about which tasks use which context flags

    """
    context_sources = CONTEXT_PRETTY_NAMES.keys()
    task_info = []
    for task_cls in tasks:
        tmp = {}
        tmp["task"] = task_cls.__name__
        tmp.update(
            {source: source in task_cls._context_sources for source in context_sources}
        )
        task_info.append(tmp)

    return (
        pd.DataFrame(task_info).set_index("task").rename(columns=CONTEXT_PRETTY_NAMES)
    )


def get_task_skill_info(tasks, omit=[]):
    """
    Get a DataFrame with information about which tasks evaluate which skill

    """
    task_info = []
    for task_cls in tasks:
        tmp = {}
        tmp["task"] = task_cls.__name__
        tmp.update(
            {
                skill.capitalize(): skill in task_cls._skills
                for skill in ALLOWED_SKILLS
                if skill not in omit
            }
        )
        task_info.append(tmp)

    return pd.DataFrame(task_info).set_index("task")


def list_tasks_per_column(task_info):
    """
    Produce an HTML list where each column is a heading and the tasks
    with a True value for that column are listed under it as a bullet list

    """
    # Initialize the HTML string
    html = ""

    # Loop over the context types
    for col in task_info.columns:
        # Add the context type as a heading
        html += f"<h3 class='text-primary'>{col}</h3>\n"

        # Get the tasks using the context type
        tasks = task_info[task_info[col]].index

        if len(tasks) == 0:
            html += "<p>No tasks</p>\n"
        else:
            # Add the tasks as a bullet list
            html += "<ul class='list-group'>\n"
            for task in tasks:
                html += f"<li class='list-group-item'><a href='./{task}.html'>{task}</a></li>\n"
            html += "</ul>\n"

    return html


def create_task_summary_page(tasks):
    """
    Create a summary page for the benchmark tasks

    """
    summary = """
<!DOCTYPE html>
<html>
<head>
    <title>Task - {task_name}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            background-color: #f8f9fa;
        }}
        .container {{
            margin-top: 20px;
        }}
        .section {{
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 0.25rem;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .section h2 {{
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        .list-group-item {{
            background-color: #f8f9fa;
        }}
        .back-button {{
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="section">
            <h1 class="text-center text-primary">{task_name}</h1>
        </div>

        <!-- Back Button -->
        <button class="btn btn-secondary back-button" onclick="history.back()">Go Back</button>

        <!-- Content -->
        {seeds}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

    context_source_mapping = {
        "c_h": "Historical information",
        "c_f": "Future information",
        "c_i": "Intemporal information",
        "c_cov": "Covariate information",
        "c_causal": "Causal information",
    }

    skill_name_map = {
        "instruction following": "Instruction Following",
        "retrieval: context": "Retrieval: Context",
        "retrieval: memory": "Retrieval: Memory",
        "reasoning: deduction": "Reasoning: Deduction",
        "reasoning: analogy": "Reasoning: Analogy",
        "reasoning: math": "Reasoning: Math",
        "reasoning: causal": "Reasoning: Causal",
    }

    for task_cls in tasks:
        task_name = task_cls.__name__
        seeds = ""
        for seed in range(1, 6):
            task = task_cls(seed=seed)

            context_sources = list(
                map(lambda x: context_source_mapping.get(x), task._context_sources)
            )
            capabilities = list(map(lambda x: skill_name_map.get(x), task._skills[2:]))

            if task.background:
                task.background = task.background.replace("\n", "<br />")
            if task.constraints:
                task.constraints = task.constraints.replace("\n", "<br />")
            if task.scenario:
                task.scenario = task.scenario.replace("\n", "<br />")

            seeds += f"""
                <div class="section">
                    <h2 class="text-center text-primary">Seed {seed}</h2>
            """

            if task.background:
                seeds += f"""
                    <p class="text-center">{task.background}</p>
                """

            if task.scenario:
                seeds += f"""
                    <p class="text-center">{task.scenario}</p>
                """

            if task.constraints:
                seeds += f"""
                    <p class="text-center"><strong>Constraints:</strong> {task.constraints}</p>
                    <br>
                """

            seeds += f"""
            <p class="text-center"><strong>Types of context:</strong> {context_sources}</p>
            <p class="text-center"><strong>Capabilities:</strong> {capabilities}</p>
            <br>
            <p class="text-center">
                {_figure_to_html(task.plot())}
            </p>
        </div>
    """
        with open(f"{task_name}.html", "w") as f:
            f.write(summary.format(task_name=task_name, seeds=seeds))


def plot_task_heatmap(task_info, plot_topic="Context Type"):
    """
    Render the task info matrix as a heatmap using Plotly.
    """
    # Convert boolean values to integers
    task_info_int = task_info.astype(int)

    # Check if all values are zero
    if (task_info_int.values == 0).all():
        colorscale = [[0, "white"], [1, "white"]]
    else:
        colorscale = [[0, "white"], [1, "red"]]

    # Calculate the height of the plot
    # Set a base height and add extra height per row
    base_height = 400
    row_height = 20  # Adjust this value as needed
    plot_height = base_height + row_height * len(task_info_int.index)

    # Create a heatmap using Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=task_info_int.values,
            x=task_info.columns,
            y=task_info.index,
            colorscale=colorscale,
            showscale=False,
        )
    )

    fig.update_layout(
        title=f"{plot_topic} by Task",
        xaxis_title=plot_topic,
        yaxis_title="Tasks",
        xaxis=dict(tickangle=-90),
        template="plotly_white",
        margin=dict(l=200, r=20, t=50, b=50),  # Adjust margins if needed
        height=plot_height,  # Set the calculated height
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def plot_task_barchart(task_info, plot_topic="Context Type"):
    """
    Plot the number of tasks using each context type using Plotly.
    """
    # Count the number of tasks using each context type
    context_counts = task_info.sum()

    # Create a bar plot using Plotly
    fig = go.Figure(
        data=[
            go.Bar(x=context_counts.index, y=context_counts.values, marker_color="blue")
        ]
    )

    fig.update_layout(
        title=f"Number of Tasks by {plot_topic}",
        xaxis_title=plot_topic,
        yaxis_title="Number of Tasks",
        template="plotly_white",
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


if __name__ == "__main__":
    tasks = ALL_TASKS
    task_context_info = get_task_context_info(tasks)
    task_skill_info = get_task_skill_info(
        tasks, omit=["forecasting", "natural language processing"]
    )
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Stats by context source
    task_by_context_bar = plot_task_barchart(
        task_context_info, plot_topic="Context Type"
    )
    task_context_heatmap = plot_task_heatmap(
        task_context_info, plot_topic="Context Type"
    )
    task_by_context_list = list_tasks_per_column(task_context_info)

    # # Stats by skill
    task_by_skill_bar = plot_task_barchart(task_skill_info, plot_topic="Skill")
    task_skill_heatmap = plot_task_heatmap(task_skill_info, plot_topic="Skill")
    task_by_skill_list = list_tasks_per_column(task_skill_info)

    create_task_summary_page(tasks)

    report = f"""
<!DOCTYPE html>
<html>
   <head>
      <title>Benchmark Task Overview</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      <link href="https://unpkg.com/gridjs/dist/theme/mermaid.min.css" rel="stylesheet" />
      <style>
         body {{
         background-color: #f8f9fa;
         }}
         .container {{
         margin-top: 20px;
         }}
         .section {{
         background-color: #ffffff;
         border: 1px solid #dee2e6;
         border-radius: 0.25rem;
         padding: 20px;
         margin-bottom: 20px;
         }}
         .subsection {{
         background-color: #f1f1f1;
         border: 1px solid #dee2e6;
         border-radius: 0.25rem;
         padding: 15px;
         margin-bottom: 15px;
         }}
         .section h2 {{
         border-bottom: 2px solid #007bff;
         padding-bottom: 10px;
         cursor: pointer;
         position: relative;
         }}
         .section h2::after {{
         content: "\\25BC";  /* Downward pointing caret */
         font-size: 1rem;
         position: absolute;
         right: 10px;
         top: 10px;
         transition: transform 0.3s ease;
         }}
         .collapse.show + h2::after {{
         transform: rotate(-180deg);  /* Rotate caret icon */
         }}
         .subsection h3 {{
         border-bottom: 1px solid #007bff;
         padding-bottom: 5px;
         cursor: pointer;
         }}
         .list-group-item {{
         background-color: #f8f9fa;
         }}
      </style>
   </head>
   <body>
      <div class="container">
         <div class="section">
            <h1 class="text-center text-primary">Context is Key: A Benchmark for Forecasting with Essential Textual Information</h1>
         </div>
         <div class="section">
            <h2 class="text-primary" data-bs-toggle="collapse" data-bs-target="#visualizations-section">Visualizations</h2>
            <div class="collapse" id="visualizations-section">
               <!-- Context Type Plots -->
               <div class="subsection">
                  <h3 class="text-secondary">Tasks by Context Type</h3>
                  <div class="mb-4">
                     {task_by_context_bar}
                  </div>
                  <div class="mb-4">
                     {task_context_heatmap}
                  </div>
               </div>
               <!-- Skill Type Plots -->
               <div class="subsection">
                  <h3 class="text-secondary">Tasks by Skill Type</h3>
                  <div class="mb-4">
                     {task_by_skill_bar}
                  </div>
                  <div class="mb-4">
                     {task_skill_heatmap}
                  </div>
               </div>
            </div>
         </div>
         <div class="section">
            <h2 class="text-primary" data-bs-toggle="collapse" data-bs-target="#tasks-by-context-section">Tasks by Context Type</h2>
            <div class="collapse" id="tasks-by-context-section">
               <div class="mb-4">
                  There is a total of {len(task_context_info)} tasks in the benchmark.
               </div>
               {task_by_context_list}
            </div>
         </div>
         <div class="section">
            <h2 class="text-primary" data-bs-toggle="collapse" data-bs-target="#tasks-by-skill-section">Tasks by Skill</h2>
            <div class="collapse" id="tasks-by-skill-section">
               <div class="mb-4">
                  There is a total of {len(task_skill_info)} tasks in the benchmark.
               </div>
               {task_by_skill_list}
            </div>
         </div>
      </div>
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
      <!-- Tables -->
      <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.0/papaparse.min.js"></script>
      <script src="https://unpkg.com/gridjs/dist/gridjs.umd.js"></script>
      <script>
         Papa.parse("https://raw.githubusercontent.com/anon-forecast/benchmark_report_dev/main/results.csv", {{
           download: true,
           complete: function(results) {{
             new gridjs.Grid({{
               columns: results.data[0].map(header => ({{
                     name: header,
               }})),
               data: results.data.slice(1), // Skip the header row
               search: true,
               sort: true,
               pagination: {{
                 enabled: true,
                 limit: 10
               }},
               fixedHeader: true,
               autoWidth: true,
               resizable: true
             }}).render(document.getElementById("result-table"));
           }}
         }});
      </script>
   </body>
</html>
"""

    with open(f"index.html", "w") as f:
        f.write(report)
