import dash
from dash import dcc, html, Input, Output, State
from pages import app1, home, overview_tab,final_dashboard,stalemate_resolution
from Calc_Sodo import *
import glob
import dash_bootstrap_components as dbc
from pages.ga_optimization_run import *


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
        dcc.Link("Project Information", href="/", style={'marginRight': '20px', 'textDecoration': 'none'}),
        " | ",
        dcc.Link("Preference & Weights", href="/preference",style={'marginRight': '20px', 'textDecoration': 'none'}),
        " | ",
        dcc.Link("Overview", href="/overview",style={'marginRight': '20px', 'textDecoration': 'none'}),
        " | ",
        dcc.Link("Dashboard", href="/dashboard",style={'marginRight': '20px', 'textDecoration': 'none'}),
        " | ",
        dcc.Link("Stalemate Resolution", href="/stalemate_resolution", style={'marginRight': '20px', 'textDecoration': 'none'})

    ], style={'padding': '10px', 'fontSize': '20px', 'textAlign': 'center'}),

    html.Hr(),

    html.Div(id="page-content", style={'margin': '20px'})
])

# Callback to switch pages
@app321.callback(
    Output("page-content", "children"), Output("current-page", "data"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/preference":
        return app1.preference_choice_layout,"/preference"
    elif pathname == "/overview":
        return overview_tab.overviewtab_layout,"/overview"
    elif pathname == "/dashboard":
        return final_dashboard.final_dashboard_layout,"/dashboard"
    elif pathname == "/stalemate_resolution":
        return stalemate_resolution.stalemate_layout,"/stalemate_resolution"
    else:
        return home.home_layout,'/home'  # Default is home page

app1.register_callbacks(app321)
overview_tab.register_callbacks(app321)
final_dashboard.register_callbacks(app321)

# Run the app
if __name__ == '__main__':
    app321.run_server(debug=True)