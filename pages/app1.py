
# For graph interaction
from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
from Calc_Sodo import *
import glob
import dash
from dash.exceptions import PreventUpdate

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
        html.Button('Write', id='write_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'}, ),
        dcc.Input(id="write_input",
                  type="text",
                  placeholder="File name",
                  style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}),
        html.Div(id='write_output', style={'marginTop': 20, "color": "green"})])
])

def register_callbacks(app):
    # PREFERENCE SETTING CALLBACKS
    @app.callback(
        Output('preference_plot', 'figure'),
        Output('manual_obj', 'value'),
        Output('manual_pref', 'value'),
        Output('end_point_1_pref', 'value'),
        Output('end_point_2_pref', 'value'),
        Input('dropdown_objective', 'value'),
        Input('set_obj_button', 'n_clicks'),
        Input('dropdown_decision_maker', 'value'),
        Input('set_end_button', 'n_clicks'),
        Input('reset_button', 'n_clicks'),
        State('current-page', 'data'),State('manual_obj', 'value'),
        State('manual_pref', 'value'),State('end_point_1_pref', 'value'),
        State('end_point_2_pref', 'value'))

    def update_graph(selected_objective, n_clicks,
                     dropdown_decision, end_n_clicks, reset_n_clicks,current_page,manual_obj,manual_pref,end1_pref,end2_pref):

        if current_page != "/preference":
            raise PreventUpdate  # Prevent callback if we're not on the "Choice" page

        #manual_obj = dash.callback_context.states.get("manual_obj.value")
        #manual_pref = dash.callback_context.states.get("manual_pref.value")
        #end1_pref = dash.callback_context.states.get("end_point_1_pref.value")
        #end2_pref = dash.callback_context.states.get("end_point_2_pref.value")
        # Get the min and max points based on the selected objective

        min_point, max_point = obj_overview.loc[selected_objective]['Min'], obj_overview.loc[selected_objective]['Max']

        global fig
        global total_preference
        global first_run

        if type(first_run) == int:
            first_run = [0, dropdown_decision, selected_objective]

        if dropdown_decision != first_run[1] or selected_objective != first_run[2]:
            first_run = [0, dropdown_decision, selected_objective]

        if total_preference[dropdown_decision][selected_objective] != None:

            fig.update_layout(
                title=f'{selected_objective} Preference')

            fig.data[0]['x'] = np.array(total_preference[dropdown_decision][selected_objective][0])
            fig.data[0]['y'] = np.array(total_preference[dropdown_decision][selected_objective][1])

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

    @app.callback(
        Output('save_output', 'children'),
        Input('save_button', 'n_clicks'),
        State('current-page', 'data'),
        State('dropdown_decision_maker','value'),
        State('dropdown_objective','value'))

    def save_preference_curve(n_clicks_save,current_page,dropdown_decision_save,dropdown_objective_save):

        if current_page != "/preference":
            raise PreventUpdate  # Prevent callback if we're not on the "Choice" page

        #dropdown_decision_save = dash.callback_context.states.get("dropdown_decision_maker.value")
        #dropdown_objective_save = dash.callback_context.states.get("dropdown_objective.value")

        global fig
        global total_preference

        if n_clicks_save > 0:
            # print(total_preference[dropdown_decision][dropdown_objective])
            x_values = list(fig.data[0]['x'])
            y_values = list(fig.data[0]['y'])
            total_preference[dropdown_decision_save][dropdown_objective_save] = [x_values, y_values]
            return f'Preference saved! {total_preference[dropdown_decision_save][dropdown_objective_save]}'
        return

    @app.callback(
        Output('reset_output', 'children'),
        Input('reset_button', 'n_clicks'),
        State('current-page', 'data'),State('dropdown_decision_maker','value'),State('dropdown_objective','value'))

    def reset_preference_curve(n_clicks_res,current_page,dropdown_decision_res,dropdown_objective_res):
        if current_page != "/preference":
            raise PreventUpdate  # Prevent callback if we're not on the "Choice"

        #dropdown_decision_res = dash.callback_context.states.get("dropdown_decision_maker.value")
        #dropdown_objective_res = dash.callback_context.states.get("dropdown_objective.value")

        global total_preference

        if n_clicks_res > 0:
            total_preference[dropdown_decision_res][dropdown_objective_res] = None
            return f'Preference curve reset for {dropdown_decision_res}, {dropdown_objective_res}.'
        return

    @app.callback(
        Output('load_output', 'children'),
        Input('load_button', 'n_clicks'),
        State('current-page', 'data'),
        State('load_dropdown', 'value'))

    def load_preference_curve(n_clicks_load,current_page,file_name):

        if current_page != "/preference":
            raise PreventUpdate  # Prevent callback if we're not on the "Choice" page

        global total_preference
        #file_name = dash.callback_context.states.get("load_dropdown.value")
        if n_clicks_load > 0 and file_name is not None:
            # total_preference = np.load(direct + file_name, allow_pickle='TRUE').item()
            total_preference = np.load(file_name, allow_pickle=True).item()
            return f'Preference file {file_name} loaded.'
        else:
            return f'First run, button counter = {n_clicks_load},selected option is {file_name},{current_page}'

    @app.callback(
        Output('write_output', 'children'),
         Output('load_dropdown', 'options'),
        Input('write_button', 'n_clicks'),
        State('write_input', 'value'),State('current-page', 'data'))
    def save_file(n_clicks_wr, file_name,current_page):
        '''
        if current_page != "/preference":
            raise PreventUpdate  # Prevent callback if we're not on the "Choice" page
        '''
        global total_preference
        global npy_files

        if n_clicks_wr > 0:
            np.save(file_name + '.npy', total_preference)
            npy_files = glob.glob("*.npy")
            return [f'Saved all preference curves as {file_name}.npy', npy_files]
        else:
            # Return default values when the button hasn't been clicked
            return "", npy_files  # Empty string for 'write_output.children', current npy_files for 'load_dropdown.options'

    return