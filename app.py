import dash
from dash import dcc, html, Input, Output, State
from pages import app1, home, weight_table_app
from ObjFun import *
from Calc_Sodo import *
import numpy as np
import glob
import plotly.graph_objects as go
import json
from pages import final_dashboard
from ga_optimization_run import preferendus_go,setup_for_display,plot_polar_ppi
from plotly.subplots import make_subplots
from scipy.interpolate import pchip_interpolate
from genetic_algorithm_pfm import GeneticAlgorithm

from pages.ga_optimization_run import cons_ga

decision_makers = dec_mak()
objective_names = obj_names()

obj_overview, total_preference = get_SODO_stuff()
npy_files = glob.glob("*.npy")

first_run = 0
#direct = '/home/ossDSS/saved_prefs/'
#npy_files = glob.glob(direct + "*.npy")
#npy_files = [s.replace(direct,"").strip() for s in npy_files]
#npy_files = glob.glob("*.npy")

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
        dcc.Link("Table", href="/table"),
        " | ",
        dcc.Link("Dashboard", href="/dashboard")
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
    elif pathname == "/dashboard":
        return final_dashboard.final_dashboard_layout
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
    flat_list = weight_table_app.table_data_to_flat_list(data)

    # Save the flat list to a file
    with open('saved_data.json', "w") as file:
        json.dump(flat_list, file)

    return html.Div([
        html.H4("Saved Data (Flat List):"),
        html.Pre(str(flat_list))
    ])

#FINAL DASHBOARD CALLBACKS

@app321.callback(Output('radial_preference_graph', 'figure'),
                 Output('opt_run_output','children'),
                 Input('run_opt_button', 'n_clicks'),
                 State('final_load_npy_dropdown','value'),
                 State('final_load_json_dropdown','value'))

def opt_graph(n_clicks_radial,npy_file_name,json_file_name):
    global result_df
    global final_weights

    if n_clicks_radial > 0:

        final_pref = np.load(npy_file_name, allow_pickle='TRUE').item()

        f = open(json_file_name,'r')

        f_loaded = json.load(f)

        undivided = [i for i in f_loaded if i != 0]

        final_weights = [x / (len(decision_makers)) for x in undivided]

        #print(final_weights)

        bounds, cons_ga = bounds_cons()

        def objective(variables):
            """
            Objective function that is fed to the GA. Calls the separate preference functions that are declared above.

            :param variables: array with design variable values per member of the population. Can be split by using array
            slicing
            :return: 1D-array with aggregated preference scores for the members of the population.
            """
            x1 = variables[:, 0]  # Distance to city centre (Oss)
            x2 = variables[:, 1]  # Distance to Den Bosch
            x3 = variables[:, 2]  # Number of turbines Oss
            x4 = variables[:, 3]  # Number of turbines Den Bosch
            x5 = variables[:, 4]  # Turbine hub height Oss
            x6 = variables[:, 5]  # Turbine hub height Den Bosch

            func = 0
            pref_all_p = []
            for i in final_pref:
                for objective in objective_names:
                    if final_pref[i][objective] != None:

                        if objective == 'NPV':
                            func = -1 * obj_NPV_ga(x1, x2, x3, x4, x5, x6)
                        elif objective == 'Noise - Oss':
                            func = obj_noise_disturbance_oss_ga(x1, x2, x3, x4, x5, x6)
                        elif objective == 'Noise - Den Bosch':
                            func = obj_noise_disturbance_bosch_ga(x1, x2, x3, x4, x5, x6)
                        elif objective == 'Bird Mortality':
                            func = obj_bird_mortality_ga(x1, x2, x3, x4, x5, x6)
                        elif objective == 'Particle Pollution':
                            func = obj_particle_pollution_ga(x1, x2, x3, x4, x5, x6)
                        elif objective == 'Energy - Oss':
                            func = -1 * obj_energy_oss_ga(x1, x2, x3, x4, x5, x6)
                        elif objective == 'Energy - Den Bosch':
                            func = -1 * obj_energy_bosch_ga(x1, x2, x3, x4, x5, x6)
                        elif objective == 'Project Time':
                            func = obj_project_time_ga(x1, x2, x3, x4, x5, x6)

                        p_temp = pchip_interpolate(list(final_pref[i][objective][0]), list(final_pref[i][objective][1]),
                                                   func)
                        pref_all_p.append(p_temp)

            # aggregate preference scores and return this to the GA
            return final_weights, pref_all_p

        def preferendus_go(final_pref, objective, bounds, cons_ga):

            # specify the number of runs of the optimization
            n_runs = 5

            # make dictionary with parameter settings for the GA
            # print('Run IMAP')
            options = {
                'n_bits': 24,
                'n_iter': 400,
                'n_pop': 1000,
                'r_cross': 0.8,
                'max_stall': 10,
                'aggregation': 'a_fine',
                'var_type_mixed': ['real', 'real', 'int', 'int', 'real', 'real']
            }

            save_array = list()  # list to save the results from every run to
            pref_array = list()
            pref_long_array = list()
            pref_loop = list()
            pref_long_obj = list()
            pref_obj = list()
            ga = GeneticAlgorithm(objective=objective, constraints=cons_ga, bounds=bounds,
                                  options=options)  # initialize GA

            # run the GA
            for i in range(n_runs):
                score, design_variables, plot_array = ga.run()
                for b in final_pref:
                    for obj in objective_names:
                        if final_pref[b][obj] != None:

                            if obj == 'NPV':
                                func = -obj_NPV(design_variables)
                            elif obj == 'Noise - Oss':
                                func = obj_noise_disturbance_oss(design_variables)
                            elif obj == 'Noise - Den Bosch':
                                func = obj_noise_disturbance_bosch(design_variables)
                            elif obj == 'Bird Mortality':
                                func = obj_bird_mortality(design_variables)
                            elif obj == 'Particle Pollution':
                                func = obj_particle_pollution(design_variables)
                            elif obj == 'Energy - Oss':
                                func = -obj_energy_oss(design_variables)
                            elif obj == 'Energy - Den Bosch':
                                func = -obj_energy_bosch(design_variables)
                            elif obj == 'Project Time':
                                func = obj_project_time(design_variables)

                            p_temp = pchip_interpolate(list(final_pref[b][obj][0]), list(final_pref[b][obj][1]), func)
                            pref_loop.append(p_temp)
                            pref_obj.append(func)

                save_array.append([design_variables[0], design_variables[1], design_variables[2], design_variables[3],
                                   design_variables[4], design_variables[5]])
                pref_long_array.append([pref_loop])
                pref_long_obj.append([pref_obj])
                pref_t = np.sum(np.multiply(final_weights, pref_loop))
                pref_array.append(pref_t)
                pref_loop = list()
                pref_obj = list()

                # print(f'The objective values are then: NPV = {-round(obj_NPV(design_variables),2)}, noise = {round(obj_noise_disturbance(design_variables),2)}, bird mortality = {round(obj_bird_mortality(design_variables),2)},particle pollution = {round(obj_particle_pollution(design_variables),2)}')
                # Back.YELLOW +
                # print(Back.RESET + f'The overall preference is {a_fine_aggregator([w1,w2,w3,w4],[pref1,pref2,pref3,pref4])}')

            pref_f_list = pref_long_array[pref_array.index(max(pref_array))]
            pref_final_all = np.array(pref_f_list)
            res = save_array[pref_array.index(max(pref_array))]
            obj_val = pref_long_obj[pref_array.index(max(pref_array))]

            return pref_final_all, res, obj_val

        pref_final_all, res, obj_val = preferendus_go(final_pref,objective,bounds,cons_ga)

        labels, result_df = setup_for_display(final_pref, objective_names, obj_val, pref_final_all)

        fig = plot_polar_ppi(pref_final_all[0], labels)

        return fig,'Optimization Performed'
    return

@app321.callback(
    Output('preference_plot_final', 'figure'),
    Input('dropdown_choice_o', 'value'),
    Input('dropdown_choice_d', 'value'),
    State('final_load_npy_dropdown','value')
)
def plot_preference(obj_to_plot, dm_to_plot,npy_file_name):
    n = 0
    final_pref = np.load(npy_file_name, allow_pickle='TRUE').item()
    sub_titles = []
    for b in final_pref:
        if b == dm_to_plot or dm_to_plot == 'All':
            for obj in objective_names:
                if obj == obj_to_plot or obj_to_plot == 'All':
                    if final_pref[b][obj] != None:
                        sub_titles.append(f'{obj},{b}')
                        n += 1
    subplot_height = 500  # Height in pixels for each subplot
    total_height = subplot_height * max(n, 1)  # Total height based on the number of subplots

    fig_p = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=sub_titles,
        vertical_spacing=0.025)  # Adjust vertical spacing as needed
    n = 1
    for b in final_pref:
        if b == dm_to_plot or dm_to_plot == 'All':
            for obj in objective_names:
                if obj == obj_to_plot or obj_to_plot == 'All':
                    if final_pref[b][obj] != None:
                        marker_x = result_df.loc[b + ',' + obj]['Objective Value']
                        marker_y = result_df.loc[b + ',' + obj]['Preference Score']

                        fig_p.add_trace(go.Scatter(x=final_pref[b][obj][0], y=final_pref[b][obj][1], name=f'{obj},{b}'),
                                        row=n, col=1)
                        fig_p.add_trace(go.Scatter(x=[marker_x], y=[marker_y], mode='markers', marker_symbol='circle',
                                                   marker_size=10, name='IMAP Solution',
                                                   marker_color='green' if marker_y > 0 else 'red'),
                                        row=n, col=1)
                        n += 1
    fig_p.update_layout(
        height=total_height,
        title="Preference Plot",
        showlegend=False)
    return fig_p

# Run the app
if __name__ == '__main__':
    app321.run_server(debug=True)