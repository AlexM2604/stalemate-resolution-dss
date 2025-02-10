from dash import Dash, html, dcc, Input, Output, State, callback
import dash
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ga_optimization_run import *
import glob
from dash.exceptions import PreventUpdate
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
        dcc.Loading(  # Adds a spinner while loading
            id="loading-1",
            type="circle",  # Choose "default", "dot", or "circle"
            children=html.Div(id="opt_run_output"),style={'padding': '10px', 'fontSize': '20px', 'textAlign': 'center'}),
        #html.Div(id="opt_run_output", style={"textAlign": "center", "marginTop": "20px"})]),

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
        html.Div(id='manual_benchmark_output', style={'marginTop': 20})])])])

def register_callbacks(app):

    @app.callback(Output('radial_preference_graph', 'figure'),
                     Output('opt_run_output', 'children'),
                     Input('run_opt_button', 'n_clicks'),
                     State('final_load_npy_dropdown', 'value'),
                     State('final_load_json_dropdown', 'value'))
    def opt_graph(n_clicks_radial, npy_file_name, json_file_name):

        if not dash.callback_context.triggered:
            raise PreventUpdate

        global result_df
        global final_weights

        if n_clicks_radial > 0:

            final_pref = np.load(npy_file_name, allow_pickle='TRUE').item()

            f = open(json_file_name, 'r')

            f_loaded = json.load(f)

            undivided = [i for i in f_loaded if i != 0]

            final_weights = [x / (len(decision_makers)) for x in undivided]

            # print(final_weights)

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

                            p_temp = pchip_interpolate(list(final_pref[i][objective][0]),
                                                       list(final_pref[i][objective][1]),
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

                                p_temp = pchip_interpolate(list(final_pref[b][obj][0]), list(final_pref[b][obj][1]),
                                                           func)
                                pref_loop.append(p_temp)
                                pref_obj.append(func)

                    save_array.append(
                        [design_variables[0], design_variables[1], design_variables[2], design_variables[3],
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

            pref_final_all, res, obj_val = preferendus_go(final_pref, objective, bounds, cons_ga)

            labels, result_df = setup_for_display(final_pref, objective_names, obj_val, pref_final_all)

            figure_polar = plot_polar_ppi(pref_final_all[0], labels)

            return figure_polar, 'Optimization Performed'
        return

    @app.callback(
        Output('preference_plot_final', 'figure'),
        Input('dropdown_choice_o', 'value'),
        Input('dropdown_choice_d', 'value'),
        State('final_load_npy_dropdown', 'value')
    )
    def plot_preference(obj_to_plot, dm_to_plot, npy_file_name):

        if not dash.callback_context.triggered:
            raise PreventUpdate

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

                            fig_p.add_trace(
                                go.Scatter(x=final_pref[b][obj][0], y=final_pref[b][obj][1], name=f'{obj},{b}'),
                                row=n, col=1)
                            fig_p.add_trace(
                                go.Scatter(x=[marker_x], y=[marker_y], mode='markers', marker_symbol='circle',
                                           marker_size=10, name='IMAP Solution',
                                           marker_color='green' if marker_y > 0 else 'red'),
                                row=n, col=1)
                            n += 1
        fig_p.update_layout(
            height=total_height,
            title="Preference Plot",
            showlegend=False)
        return fig_p

        return

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
