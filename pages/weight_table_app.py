import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import json
import os
from Calc_Sodo import obj_names,dec_mak



# File to save the table data
save_file = "saved_data.json"

objective_names = obj_names()
decision_makers = dec_mak()

def load_saved_data():
    if os.path.exists(save_file):
        with open(save_file, "r") as file:
            return json.load(file)
    else:
        # Default data if no saved file exists
        return [0.0] * (len(objective_names) * len(decision_makers))  # Flat list of zeros

# Load data when the app starts
initial_data = load_saved_data()

# Convert flat list to table data format
def flat_list_to_table_data(flat_list):
    table_data = []
    num_objectives = len(objective_names)
    for i, dm in enumerate(decision_makers):
        start_index = i * num_objectives
        end_index = start_index + num_objectives
        row_values = flat_list[start_index:end_index]
        row_dict = {"Row Title": dm, **{obj: val for obj, val in zip(objective_names, row_values)}}
        table_data.append(row_dict)
    return table_data

# Convert table data to flat list
def table_data_to_flat_list(table_data):
    flat_list = []
    for row in table_data:
        flat_list.extend([float(row[obj]) for obj in objective_names])
    return flat_list

# Initialize table data
table_data = flat_list_to_table_data(initial_data)

# Dash layout
tableapp_layout = html.Div([
    html.H1("Objective Weights per Decision-Maker", style={"textAlign": "center"}),

    dash_table.DataTable(
        id="editable-table",
        columns=[
            {"name": "Row Title", "id": "Row Title", "editable": False}
        ] + [
            {"name": col, "id": col, "editable": True if col != "Sum" else False}
            for col in objective_names + ["Sum"]
        ],
        data=table_data,
        style_data_conditional=[],
        style_table={"margin": "auto", "width": "60%"},
        style_cell={
            "textAlign": "center",
            "fontFamily": "Arial",
        },
        style_header={"fontWeight": "bold"},
    ),

    html.Button("Save Table Data", id="save_button_table", style={"marginTop": "50px", "display": "block", "margin": "auto"}),

    html.Div(id="save_table_output", style={"textAlign": "center", "marginTop": "20px"}),
])