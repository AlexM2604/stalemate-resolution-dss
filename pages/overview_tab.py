import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import json
import os
from Calc_Sodo import obj_names,dec_mak
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go

objective_names = obj_names()
decision_makers = dec_mak()

def weight_save_name():
    names_of_saves = {'Energy Provider':'EP','Local Residents - Oss':'LRO', 'Local Residents - Den Bosch':'LRDB', 'Ecologists':'ECO',
                       'RIVM':'RIVM', 'Oss Municipality':'OM', 'Den Bosch Municipality':'DBM'}
    return names_of_saves

def get_initial_table():
    save_names = weight_save_name()
    all_the_weights = []
    for a in save_names:
        if os.path.exists(save_names[a] + '.json'):
            with open(save_names[a] + '.json', "r") as file:
                weight = json.load(file)
                all_the_weights += weight
    return all_the_weights

initial_table_data = get_initial_table()
#print(initial_table_data)

overviewtab_layout = html.Div(children = [

    html.H1(children='Overview of All Decision-Makers'),

    html.Div(children=[
        html.H2(children='Objective Weights'),
        dash_table.DataTable(data = initial_table_data, editable = True, id="overview_table",
                         style_table={'width': '50%','marginLeft': 100, 'marginRight': 100,'margin-top': '20px','margin-bottom': '20px'},
                         style_cell={'textAlign': 'center','fontFamily': 'Arial'},
                         style_header={'fontWeight': 'bold'}),
        html.Button("Refresh Table", id="refresh_table",n_clicks=0, style={"marginTop": "50px", "display": "block", "margin": "auto"})
    ]),

    html.Div(children=[
        html.H2(children='Preference Curves'),
        html.Button("Refresh Curves", id="refresh_curves",n_clicks=0, style={"marginTop": "50px", "display": "block", "margin": "auto"}),
        html.Label('Choose Objective:'),
        dcc.Dropdown(objective_names + ['All'], 'All', id='dropdown_o_ov'),
        html.Label('Choose Decision-Maker:'),
        dcc.Dropdown(decision_makers + ['All'], 'All', id='dropdown_d_ov'),
        dcc.Graph(id='dm_curves')
    ])

],style={'marginLeft': 100, 'marginRight': 100})

def register_callbacks(app):
    @app.callback(
        Output('overview_table','data'),
        Input('refresh_table','n_clicks'))

    def update_ovrw_table(n_clicks):

        upd = get_initial_table()

        return upd

    @app.callback(
        Output('dm_curves','figure'),
        Input('refresh_curves','n_clicks'),
        Input('dropdown_o_ov','value'),
        Input('dropdown_d_ov', 'value')
    )
    def plot_preference(n_clicks,obj_to_plot, dm_to_plot):


        save_names = weight_save_name()
        final_pref = {}
        n = 0
        for b in decision_makers:
            if os.path.exists(save_names[b] + '_' + 'final.npy'):
                pref = np.load(save_names[b] + '_' + 'final.npy',allow_pickle='TRUE').item()
                final_pref[b] = pref

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

                            fig_p.add_trace(
                                go.Scatter(x=final_pref[b][obj][0], y=final_pref[b][obj][1], name=f'{obj},{b}'),
                                row=n, col=1)
                            n += 1
        fig_p.update_layout(
            height=total_height,
            title="Preference Plot",
            showlegend=False)

        return fig_p