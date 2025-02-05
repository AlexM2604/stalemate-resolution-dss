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

# Function to load saved data if it exists
def load_saved_data():
    if os.path.exists(save_file):
        with open(save_file, "r") as file:
            return json.load(file)
    else:
        # Default data if no saved file exists
        return [
            {objective: 0 for objective in objective_names}
            for _ in range(len(decision_makers))
        ]

# Load data when the app starts
initial_data = load_saved_data()

# Updated column and row titles
columns = ["Row Title"] + objective_names + ["Sum"]
row_titles = decision_makers

table_data = [
    {"Row Title": row_titles[i], **initial_data[i]} for i in range(len(row_titles))
]

# Dash layout
tableapp_layout = html.Div([
    html.H1("Objective Weights per Decision-Maker", style={"textAlign": "center"}),

    dash_table.DataTable(
        id="editable-table",
        columns=[
            {"name": "Row Title", "id": "Row Title", "editable": False}
        ] + [
            {"name": col, "id": col, "editable": True if col != "Sum" else False}
            for col in columns[1:]
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




'''
# Run the Dash app
if __name__ == "__main__":
    app_table.run_server(debug=True)
'''