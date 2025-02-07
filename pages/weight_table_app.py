import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import json
import os
from Calc_Sodo import obj_names,dec_mak
from dash.exceptions import PreventUpdate

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

def register_callbacks(app):
    @app.callback(
        [Output("editable-table", "data"),
         Output("editable-table", "style_data_conditional")],
        [Input("editable-table", "data_previous")],
        [State("editable-table", "data")],
    )
    def update_table(data_previous, data):

        if not dash.callback_context.triggered:
            raise PreventUpdate

        if data_previous is None:
            raise dash.exceptions.PreventUpdate

        updated_data = []
        style_data_conditional = []
        for row in data:
            # Calculate the sum for the row
            try:
                row_sum = sum(
                    float(row[col]) for col in objective_names
                )
            except ValueError:
                row_sum = 0

            # Update the "Sum" column
            row["Sum"] = round(row_sum, 2)

            # Append to updated data
            updated_data.append(row)

            # Apply conditional formatting to the "Sum" column
            color = "green" if row_sum == 1 else "red"
            style_data_conditional.append({
                "if": {"filter_query": f"{{Row Title}} = '{row['Row Title']}'", "column_id": "Sum"},
                "color": color,
                "fontWeight": "bold"
            })

        return updated_data, style_data_conditional

    # Callback to save the table data as a flat list and persist it to a file
    @app.callback(
        Output("save_table_output", "children"),
        [Input("save_button_table", "n_clicks")],
        [State("editable-table", "data")],
    )
    def save_data(n_clicks, data):

        if not dash.callback_context.triggered:
            raise PreventUpdate

        if n_clicks is None:
            raise dash.exceptions.PreventUpdate

        # Convert table data to a flat list of floats
        flat_list = table_data_to_flat_list(data)

        # Save the flat list to a file
        with open('saved_data.json', "w") as file:
            json.dump(flat_list, file)

        return html.Div([
            html.H4("Saved Data (Flat List):"),
            html.Pre(str(flat_list))
        ])

        return