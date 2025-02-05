import dash
from dash import dcc, html, Input, Output, State
from pages import app1, home, weight_table_app
from ObjFun import *
from Calc_Sodo import *
import numpy as np
import glob
import plotly.graph_objects as go
import json


decision_makers = dec_mak()
objective_names = obj_names()

obj_overview, total_preference = get_SODO_stuff()

first_run = 0
#direct = '/home/ossDSS/saved_prefs/'
#npy_files = glob.glob(direct + "*.npy")
#npy_files = [s.replace(direct,"").strip() for s in npy_files]
npy_files = glob.glob("*.npy")
# Initialize the Dash app
app321 = dash.Dash(__name__,suppress_callback_exceptions=True)

# Expose Flask server instance for WSGI
#server = app321.server

app321.title = "Multi-Page Dash App"

# Define main layout with navigation
app321.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div([
        dcc.Link("Home", href="/"),
        " | ",
        dcc.Link("Choice", href="/choice"),
        " | ",
        dcc.Link("Table", href="/table")
    ], style={'padding': '10px', 'fontSize': '20px'}),

    html.Div(id="page-content")  # Content will be loaded here
])

# Callback to switch pages
@app321.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/choice":
        return app1.preference_choice_layout
    elif pathname == "/table":
        return weight_table_app.tableapp_layout
    else:
        return home.home_layout  # Default is home page

#PREFERENCE SETTING CALLBACKS
@app321.callback(
    Output('preference_plot', 'figure'),
    Output('manual_obj', 'value'),
    Output('manual_pref', 'value'),
    Output('end_point_1_pref', 'value'),
    Output('end_point_2_pref', 'value'),
    Input('dropdown_objective', 'value'),
    Input('set_obj_button', 'n_clicks'),
    State('manual_obj', 'value'), State('manual_pref', 'value'), Input('dropdown_decision_maker', 'value'),
    Input('set_end_button', 'n_clicks'), State('end_point_1_pref', 'value'), State('end_point_2_pref', 'value'),
    Input('reset_button', 'n_clicks'))

def update_graph(selected_objective, n_clicks, manual_obj, manual_pref,
                 dropdown_decision_maker, end_n_clicks, end1_pref, end2_pref, reset_n_clicks):
    # Get the min and max points based on the selected objective
    min_point, max_point = obj_overview.loc[selected_objective]['Min'], obj_overview.loc[selected_objective]['Max']

    global fig
    global total_preference
    global first_run

    if type(first_run) == int:
        first_run = [0, dropdown_decision_maker, selected_objective]

    if dropdown_decision_maker != first_run[1] or selected_objective != first_run[2]:
        first_run = [0, dropdown_decision_maker, selected_objective]

    if total_preference[dropdown_decision_maker][selected_objective] != None:

        fig.update_layout(
            title=f'{selected_objective} Preference')

        fig.data[0]['x'] = np.array(total_preference[dropdown_decision_maker][selected_objective][0])
        fig.data[0]['y'] = np.array(total_preference[dropdown_decision_maker][selected_objective][1])

    elif first_run[0] == 0:

        # Create a plotly graph
        fig = go.Figure(layout_yaxis_range=[0, 100])

        # Add two points (min and max) as scatter points

        fig.add_trace(go.Scatter(
            x=[min_point, max_point],
            y=[0, 0],
            mode='markers+lines',
            marker=dict(size=10),
            name=selected_objective
        ))

        # Update layout for the figure
        fig.update_layout(
            title=f'{selected_objective} Preference',
            xaxis_title='Objective Value',  # Placeholder for x-axis, can change based on your use case
            yaxis_title='Preference Score',
            showlegend=True)
        first_run[0] = 1

    if n_clicks > 0 and manual_obj is not None and manual_pref is not None:
        new_point = [manual_obj, manual_pref]

        x_t = list(fig.data[0]['x'])
        y_t = list(fig.data[0]['y'])
        x_t.append(new_point[0])
        y_t.append(new_point[1])
        y_new = [y for x, y in sorted(zip(x_t, y_t))]
        x_t.sort()

        fig.data[0]['x'] = np.array(x_t)
        fig.data[0]['y'] = np.array(y_new)

        return fig, None, None, None, None

    if end_n_clicks > 0 and end1_pref is not None and end2_pref is not None:
        new_end_obj, new_end_pref = [min_point, max_point], [end1_pref, end2_pref]

        fig.data[0]['x'] = np.array(new_end_obj)
        fig.data[0]['y'] = np.array(new_end_pref)

        return fig, None, None, None, None

    return fig, None, None, None, None

@app321.callback(
    Output('save_output', 'children'),
    Input('save_button', 'n_clicks'),
    State('dropdown_decision_maker', 'value'),
    State('dropdown_objective', 'value'))
def save_preference_curve(n_clicks_save, dropdown_decision_maker_save, dropdown_objective_save):
    global fig
    global total_preference

    if n_clicks_save > 0:
        # print(total_preference[dropdown_decision_maker][dropdown_objective])
        x_values = list(fig.data[0]['x'])
        y_values = list(fig.data[0]['y'])
        total_preference[dropdown_decision_maker_save][dropdown_objective_save] = [x_values, y_values]
        return f'Preference saved! {total_preference[dropdown_decision_maker_save][dropdown_objective_save]}'
    return


@app321.callback(
    Output('reset_output', 'children'),
    Input('reset_button', 'n_clicks'),
    State('dropdown_decision_maker', 'value'),
    State('dropdown_objective', 'value'))
def save_preference_curve(n_clicks_res, dropdown_decision_maker_res, dropdown_objective_res):
    global total_preference

    if n_clicks_res > 0:
        total_preference[dropdown_decision_maker_res][dropdown_objective_res] = None
        return f'Preference curve reset for {dropdown_decision_maker_res}, {dropdown_objective_res}.'
    return


@app321.callback(
    Output('load_output', 'children'),
    Input('load_button', 'n_clicks'),
    State('load_dropdown', 'value'))
def load_preference_curve(n_clicks_load, file_name):
    global total_preference

    if n_clicks_load > 0:
        #total_preference = np.load(direct + file_name, allow_pickle='TRUE').item()
        total_preference = np.load(file_name, allow_pickle='TRUE').item()
        return f'Preference file {file_name} loaded.'
    return


@app321.callback(
    [Output('write_output', 'children'),
     Output('load_dropdown', 'options')],
    Input('write_button', 'n_clicks'),
    State('write_input', 'value'))
def save_file(n_clicks_wr, file_name):
    global total_preference
    global npy_files

    if n_clicks_wr > 0:
        np.save(file_name + '.npy', total_preference)
        npy_files = glob.glob("*.npy")
        return [f'Saved all preference curves as {file_name}.npy', npy_files]
    else:
        # Return default values when the button hasn't been clicked
        return ["", npy_files]  # Empty string for 'write_output.children', current npy_files for 'load_dropdown.options'


# WEIGHT TABLE CALLBACKS
@app321.callback(
    [Output("editable-table", "data"),
     Output("editable-table", "style_data_conditional")],
    [Input("editable-table", "data_previous")],
    [State("editable-table", "data")],
)
def update_table(data_previous, data):
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
@app321.callback(
    Output("save_table_output", "children"),
    [Input("save_button_table", "n_clicks")],
    [State("editable-table", "data")],
)
def save_data(n_clicks, data):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Convert table data to a flat list of floats
    saved_data = []
    for row in data:
        try:
            # Extract and convert values to floats
            row_values = [float(row[col]) for col in objective_names]
            saved_data.extend(row_values)  # Add values to the flat list
        except ValueError:
            return html.Div("Error: Please enter valid numerical values in the table.")

    # Save the current table data to a file
    with open(save_file, "w") as file:
        json.dump(data, file)

    return html.Div([
        html.H4("Saved Data:"),
        html.Pre(str(saved_data))
    ])


# Run the app
if __name__ == '__main__':
    app321.run_server(debug=True)