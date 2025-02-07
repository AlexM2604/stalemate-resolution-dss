import dash
from dash import dcc, html, Input, Output, State
from pages import app1, home, weight_table_app,final_dashboard
from ObjFun import *
from Calc_Sodo import *
import numpy as np
import glob
import plotly.graph_objects as go
import json
import dash_bootstrap_components as dbc

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
app321 = dash.Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.MINTY])

# Expose Flask server instance for WSGI
#server = app321.server

app321.title = "Multi-Page Dash App"

# Define main layout with navigation
app321.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="current-page", storage_type="memory"),
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
    Output("page-content", "children"), Output("current-page", "data"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/choice":
        return app1.preference_choice_layout,"/choice"
    elif pathname == "/table":
        return weight_table_app.tableapp_layout,"/table"
    elif pathname == "/dashboard":
        return final_dashboard.final_dashboard_layout,"/dashboard"
    else:
        return home.home_layout,'/home'  # Default is home page

app1.register_callbacks(app321)
weight_table_app.register_callbacks(app321)
final_dashboard.register_callbacks(app321)

# Run the app
if __name__ == '__main__':
    app321.run_server(debug=True)