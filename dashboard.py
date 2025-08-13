import dash
from dash import html, dcc
import dash.dependencies as dd
import re
from datetime import datetime
import os
from dataset import TrainingDataset

EXCLUDE_LOGS = [
  "worker.cpp:24",
  "worker.cpp:70",
  "model_eval.cpp:171",
  "model_eval.cpp:128",
  "model_eval.cpp:156",
  "model_eval.cpp:226",
  "Batch",
  "evaluator.cpp"
]

# Updated read_log function
def read_log(file_path):
  with open(file_path, 'r') as file:
    lines = file.readlines()
  processed_lines = []
  for line in lines:
    if any(excluded in line for excluded in EXCLUDE_LOGS):
      continue
    processed_lines.append(line.strip())
  return "\n".join(processed_lines)

# Function to get the last modified file
def get_last_modified_file():
    files = [
        '/home/krkartikay/code/min-alpha-zero/model_eval.log',
        '/home/krkartikay/code/min-alpha-zero/alpha_zero.log',
        '/home/krkartikay/code/min-alpha-zero/training.log'
    ]
    last_modified_file = max(files, key=os.path.getmtime)
    return last_modified_file

# Initialize Dash app
app = dash.Dash(__name__)

# Updated layout for the dashboard
app.layout = html.Div([
  html.Div([
    html.H3("Model Evaluation Log", id='model-eval-heading'),
    html.P(id='model-eval-log-length', style={'font-size': '12px', 'color': 'gray'}),
    html.Pre(id='model-eval-log', style={'whiteSpace': 'pre-wrap', 'padding': '10px', 'font-size': '10px'})
  ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
  html.Div([
    html.H3("Alpha Zero Log", id='alpha-zero-heading'),
    html.P(id='alpha-zero-log-length', style={'font-size': '12px', 'color': 'gray'}),
    html.Pre(id='alpha-zero-log', style={'whiteSpace': 'pre-wrap', 'padding': '10px', 'font-size': '10px'})
  ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
  html.Div([
    html.H3("Training Log", id='training-heading'),
    html.P(id='training-log-length', style={'font-size': '12px', 'color': 'gray'}),
    html.Pre(id='training-log', style={'whiteSpace': 'pre-wrap', 'padding': '10px', 'font-size': '10px'})
  ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
], style={'fontFamily': 'sans-serif'})

# Function to generate heading with optional green circle styled using CSS
def generate_heading(base_text, is_last_modified):
    green_circle = html.Span("●", style={'color': 'green', 'font-size': '14px', 'margin-left': '5px'})
    return html.Span([base_text, green_circle] if is_last_modified else [base_text])

# Updated callback to include CSS-styled green circle for last modified file
@app.callback(
    [dd.Output('model-eval-log', 'children'),
     dd.Output('model-eval-log-length', 'children'),
     dd.Output('alpha-zero-log', 'children'),
     dd.Output('alpha-zero-log-length', 'children'),
     dd.Output('training-log', 'children'),
     dd.Output('training-log-length', 'children'),
     dd.Output('model-eval-heading', 'children'),
     dd.Output('alpha-zero-heading', 'children'),
     dd.Output('training-heading', 'children')],
    [dd.Input('interval-component', 'n_intervals')]
)
def update_logs(n):
    def get_original_length(file_path):
        with open(file_path, 'r') as file:
            return len(file.readlines())

    model_eval_content = read_log('/home/krkartikay/code/min-alpha-zero/model_eval.log')
    alpha_zero_content = read_log('/home/krkartikay/code/min-alpha-zero/alpha_zero.log')
    training_content = read_log('/home/krkartikay/code/min-alpha-zero/training.log')
    
    model_eval_original_length = get_original_length('/home/krkartikay/code/min-alpha-zero/model_eval.log')
    alpha_zero_original_length = get_original_length('/home/krkartikay/code/min-alpha-zero/alpha_zero.log')
    training_original_length = get_original_length('/home/krkartikay/code/min-alpha-zero/training.log')
    
    # Load dataset and get number of rows
    dataset = TrainingDataset("/home/krkartikay/code/min-alpha-zero/training_data.bin")
    num_rows = len(dataset)

    model_eval_length = f"{model_eval_original_length} lines"
    alpha_zero_length = f"{alpha_zero_original_length} lines"
    training_length = f"{training_original_length} lines | {num_rows//1000}k rows"
    
    last_modified_file = get_last_modified_file()
    model_eval_heading = generate_heading("Model Evaluation Log", last_modified_file.endswith("model_eval.log"))
    alpha_zero_heading = generate_heading("Alpha Zero Log", last_modified_file.endswith("alpha_zero.log"))
    training_heading = generate_heading("Training Log", last_modified_file.endswith("training.log"))
    
    return model_eval_content, model_eval_length, alpha_zero_content, alpha_zero_length, training_content, training_length, model_eval_heading, alpha_zero_heading, training_heading

# Interval for periodic updates
app.layout.children.append(
    dcc.Interval(
        id='interval-component',
        interval=5000,  # Update every 5 seconds
        n_intervals=0
    )
)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)