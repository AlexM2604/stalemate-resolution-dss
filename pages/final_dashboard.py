from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ga_optimization_run import *

decision_makers = dec_mak()


final_dashboard_layout = html.Div(children=[
    html.H1(children='Optimization Run and Results Display'),

    html.Div(children='Run Optimization:'),

    html.Button("Run", id="run_opt_button",
                style={"marginTop": "50px", "display": "block", "margin": "auto"}),

    html.Div(id="opt_run_output", style={"textAlign": "center", "marginTop": "20px"}),

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
        dcc.Graph(id='preference_plot')]),
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


@app321.callback(Output('radial_preference_graph','figure'),
                 Input('run_opt_button','n_clicks'))

def opt_graph():

    pref_final_all,res,obj_val = preferendus_go()

    labels, result_df = setup_for_display(final_pref,objective_names,obj_val,pref_final_all)

    fig = plot_polar_ppi(pref_final_all[0],labels)

    return fig
@app321.callback(
    Output('preference_plot', 'figure'),
    Input('dropdown_choice_o', 'value'),
    Input('dropdown_choice_d', 'value')
)
def plot_preference(obj_to_plot, dm_to_plot):
    n = 0
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


if __name__ == '__main__':
    app_oss_adjusted.run(debug=True, port=8073)