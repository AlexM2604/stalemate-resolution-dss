from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ga_optimization_run import *
import glob

from pages.ga_optimization_run import objective_names

decision_makers = dec_mak()
npy_files = glob.glob("*.npy")
json_files = glob.glob("*.json")
objective_names = obj_names()

final_dashboard_layout = html.Div(children=[
    html.H1(children='Optimization Run and Results Display'),

    html.Div(children=[
        html.Label('Load optimization files:'),
        dcc.Dropdown(npy_files, id='final_load_npy_dropdown',
                     style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}),
        dcc.Dropdown(json_files, id='final_load_json_dropdown',
                     style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}),
        html.Button("Run", id="run_opt_button",
                style={"marginTop": "50px", "display": "block", "margin": "auto"}),
        html.Div(id="opt_run_output", style={"textAlign": "center", "marginTop": "20px"})]),

    html.Div(children='''

        Insert description

    '''),

    # html.Div(children= f'Optimal project configuration is Distance from city centre = {round(res[0], 2)} km, Number of turbines = {round(res[1], 2)}, Turbine height = {round(res[2], 2)} m',
    # style={'marginTop': 20}),

    # html.Div(children= f'This results in the following objective values: profit = {-round(obj_NPV(res),2)}, noise = {round(obj_noise_disturbance(res),2)}, bird mortality = {round(obj_bird_mortality(res),2)},particle pollution = {round(obj_particle_pollution(res),2)}',
    # style={'marginTop': 20}),

    dcc.Graph(
        id='radial_preference_graph'),

    html.Div(children=[
        html.Label('Choose Objective:'),
        dcc.Dropdown(objective_names + ['All'], 'All', id='dropdown_choice_o'),
        html.Label('Choose Decision-Maker:'),
        dcc.Dropdown(decision_makers + ['All'], 'All', id='dropdown_choice_d'),  # Return and make dynamic
        dcc.Graph(id='preference_plot_final')]),
    html.Button('Reset Benchmark', id='reset_button', n_clicks=0, style={'marginTop': 20}),
    html.Div(id='reset_output', style={'marginTop': 20}),

    # Section for manual input of a benchmark
    html.Div(children=[
        html.Label('Benchmark value:'),
        dcc.Input(id='manual_benchmark', type='number', value='', placeholder='Enter value'),
        dcc.Dropdown(objective_names, objective_names[0], id='manual_dropdown', style={'marginTop': '20px'}),
        html.Button('Set Benchmark', id='set_benchmark_button', n_clicks=0,
                    style={'marginLeft': '10px', 'marginTop': '20px'}, ),
        html.Div(id='manual_benchmark_output', style={'marginTop': 20})])])

'''
# Callback to handle the "Reset Benchmark" button
@app_oss_adjusted.callback(
    Output('reset_output', 'children'),
    Input('reset_button', 'n_clicks'),
    State('dropdown_choice_o', 'value')
)
def reset_benchmark(n_clicks, selected_graph):
    global cons_ga
    if n_clicks > 0:
        # Run the set command using new_constraint
        cons_ga = add_new_constr(new_constraint, cons_ga, reset=1)
        return html.Div("Benchmark Reset", style={'color': 'green'})
    return ''


# Callback for manually setting the benchmark
@app_oss_adjusted.callback(
    Output('manual_benchmark_output', 'children'),
    [Input('set_benchmark_button', 'n_clicks')],
    [State('manual_benchmark', 'value'),
     State('manual_dropdown', 'value')]
)
def set_manual_benchmark(n_clicks, manual_input, dropdown_selection):
    global new_constraint
    global cons_ga
    if n_clicks > 0 and manual_input is not None:
        # Save the manually entered benchmark as [manual_input, 0, dropdown_selection]
        new_constraint = [manual_input, 0, dropdown_selection]
        cons_ga = add_new_constr(new_constraint, cons_ga, reset=0)
        return f'Manual benchmark set: {new_constraint}'
    return 'No manual benchmark set yet.'
'''
