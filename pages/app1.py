
# For graph interaction
from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
from Calc_Sodo import *
import glob
#from app import app321

decision_makers = dec_mak()
objective_names = obj_names()

obj_overview, total_preference = get_SODO_stuff()


first_run = 0
#direct = '/home/ossDSS/saved_prefs/'
#npy_files = glob.glob(direct + "*.npy")
npy_files = glob.glob("*.npy")
#npy_files = [s.replace(direct,"").strip() for s in npy_files]


preference_choice_layout = html.Div(children=[
    html.H1(children='Preference-Setting by Decision-Makers'),

    html.Div(children='''

    '''),

    # Section for saving the designed preference curve
    html.Div(children=[
        html.Label('Load previously saved curve:'),
        html.Button('Load', id='load_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'}),
        dcc.Dropdown(npy_files, id='load_dropdown',
                     style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}),
        html.Div(id='load_output', style={'marginTop': 5, "color": "green"})]),

    html.Div(children=[
        html.Label('Choose Decision-Maker:'),
        dcc.Dropdown(decision_makers,
                     'Energy Provider', id='dropdown_decision_maker',
                     style={'marginTop': '10px', 'marginBottom': '20px'})]),

    html.Div(children=[
        html.Label('Choose Objective:'),
        dcc.Dropdown(objective_names,
                     'NPV', id='dropdown_objective', style={'marginTop': '10px'}),
        dcc.Graph(id='preference_plot')]),

    # Option to reset preference curve

    html.Div(children=[
        html.Label('0.Reset the saved preference curve:'),
        html.Button('Reset', id='reset_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'}, ),
        html.Div(id='reset_output', style={'marginTop': 20,"color": "green"})]),

    # Section for changing end-points

    html.Div(children=[
        html.Label('1. Preference for end-point 1:'),
        dcc.Input(id='end_point_1_pref', type='number', value='', placeholder='Enter value'),
        html.Label('Preference for end-point 2:', style={'marginLeft': '10px'}),
        dcc.Input(id='end_point_2_pref', type='number', value='', placeholder='Enter value'),
        html.Button('Set end-points', id='set_end_button', n_clicks=0,
                    style={'marginLeft': '10px', 'marginTop': '20px'})]),

    # Section for manual input of preference

    html.Div(children=[
        html.Label('2. Objective value:'),
        dcc.Input(id='manual_obj', type='number', value='', placeholder='Enter value'),
        html.Label('Preference value:', style={'marginLeft': '10px'}),
        dcc.Input(id='manual_pref', type='number', value='', placeholder='Enter value'),
        html.Button('Set Point', id='set_obj_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'}, ),
        html.Div(id='manual_obj_output', style={'marginTop': '20px'})]),

    # Section for saving the designed preference curve
    html.Div(children=[
        html.Label('3. Save the preference curve (temporary, curve is not saved on program exit):'),
        html.Button('Save', id='save_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'}, ),
        html.Div(id='save_output', style={'marginTop': 20, "color": "green"})]),

    html.Div(children=[
        html.Label('4. Write all previously saved preference curves to a file (permanent save):'),
        html.Button('Write', id='write_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'}, ), \
        dcc.Input(id="write_input",
                  type="text",
                  placeholder="File name",
                  style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}),
        html.Div(id='write_output', style={'marginTop': 20, "color": "green"})])
])



