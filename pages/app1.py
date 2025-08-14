
# For graph interaction
from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
from Calc_Sodo import *
import glob
import dash
import plotly
from dash.exceptions import PreventUpdate
from dash import callback_context as ctx
from dash import dash_table
import json
import os

decision_makers = dec_mak()
objective_names = obj_names()
objective_units = units_obj()
basic_table_data = [{key: 0 for key in objective_names}]
basic_table_data[0].update({'Decision-Maker':''})
obj_overview, total_preference = get_SODO_stuff()
#print(basic_table_data)

first_run = 0
#direct = '/home/ossDSS/saved_prefs/'
#npy_files = glob.glob(direct + "*.npy")
npy_files = glob.glob("*.npy")
#npy_files = [s.replace(direct,"").strip() for s in npy_files]

def weight_save_name():
    names_of_saves = {'Energy Provider':'EP','Local Residents - Oss':'LRO', 'Local Residents - Den Bosch':'LRDB', 'Ecologists':'ECO',
                       'RIVM':'RIVM', 'Oss Municipality':'OM', 'Den Bosch Municipality':'DBM'}
    return names_of_saves

#npy_files = [s.replace(direct,"").strip() for s in npy_files]


preference_choice_layout = html.Div(children=[

    dcc.Store(id = 'preference_graph'), #For preference plotting
    dcc.Store(data = total_preference,id = 'total_preference'),#For preference plotting
    dcc.Store(data = first_run, id = 'first_run'),#For preference plotting

    html.H1(children='Preference & Weights'),

    html.Div(children=[
        html.Label('Choose Decision-Maker:'),
        dcc.Dropdown(decision_makers,
                     'Energy Provider', id='dropdown_decision_maker',
                     style={'marginTop': '10px', 'marginBottom': '20px'})]),

    html.P('''
            This section of the model allows you to view and and change the preference curves and weights for each objective,
            to express your priorities for the project. Preference is stated on a scale of -100 to 100, where 100 is the best result, 
            and -100 is the worst result (for each decision-maker). Preference in the region 0 to 100 houses desirable results.
            Preference of 0 signifies results that are acceptable, but require extra discussion. Preference below 0 signifies
            unacceptable results (or only acceptable under extreme conditions). 
    '''),

    # Section for saving the designed preference curve
    html.Div(children=[
        html.Label('Load previously saved set of curves:'),
        dcc.Dropdown(npy_files, id='load_dropdown',
                     style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}),
        html.Button('Load', id='load_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'}),
        html.Div(id='load_output', style={'marginTop': 5, "color": "green"})]),

    html.Div(children=[
        html.Label('Choose Objective:'),
        dcc.Dropdown(objective_names,
                     'NPV', id='dropdown_objective', style={'marginTop': '10px'}),
        dcc.Graph(id='preference_plot')]),

    # Option to reset preference curve

    html.Div(children=[
        html.H2(children='Preference Curve Procedure'),
        html.Label('0.Erase existing preference curve (if necessary):'),
        html.Button('Reset', id='reset_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'}, ),
        html.Div(id='reset_output', style={'marginTop': 20,"color": "green"})]),

    # Section for changing end-points

    html.Div(children=[
        html.Label('1. Set preference for minimum point (left):'),
        dcc.Input(id='end_point_1_pref', type='number', value='', placeholder='Enter Preference'),
        html.Label('and for maximum point (right):', style={'marginLeft': '10px'}),
        dcc.Input(id='end_point_2_pref', type='number', value='', placeholder='Enter Preference'),
        html.Button('Set end-points', id='set_end_button', n_clicks=0,
                    style={'marginLeft': '10px', 'marginTop': '20px'})]),

    # Section for manual input of preference

    html.Div(children=[
        html.Label('2. Set the points of interest. Objective value:'),
        dcc.Input(id='manual_obj', type='number', value='', placeholder='Enter value'),
        html.Label('Preference value:', style={'marginLeft': '10px'}),
        dcc.Input(id='manual_pref', type='number', value='', placeholder='Enter value'),
        html.Button('Set Point', id='set_obj_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'}, ),
        html.Div(id='manual_obj_output', style={'marginTop': '20px'})]),

    # Section for saving the designed preference curve
    html.Div(children=[
        html.Label('3. Save the preference curve (if you want to switch to another objective):'),
        html.Button('Save', id='save_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'}, ),
        html.Div(id='save_output', style={'marginTop': 20, "color": "green"})]),

    html.Div(children=[
        html.Label('4. Save all preferences as a file on the server. For saving this as the final preference, call it "final":'),
        dcc.Input(id="write_input",
                  type="text",
                  placeholder="File name",
                  style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}),
        html.Button('Write', id='write_button', n_clicks=0, style={'marginLeft': '10px', 'marginTop': '20px'} ),
        html.Div(id='write_output', style={'marginTop': 20, "color": "green"})]),

    html.H2(children='Weights Procedure'),
    html.P('''
            The weight of each objective means how important it is to you as a decision-maker. You have a total of 1 point to 
            allocate among all objective that you view as important for you. Please enter the weights as decimals with a point (e.g. 0.1)
            and make sure their total sum is 1. Also, you can only put weights on objectives you have defined a preference curve for.
    '''),

    dash_table.DataTable(basic_table_data, editable = True,
        id="table_app1",columns=[{"name": col, "id": col,'type': 'numeric','validation': {"allow_null": True}} for col in basic_table_data[0].keys()],
                         style_table={'width': '50%','marginLeft': 100, 'marginRight': 100,'margin-top': '20px','margin-bottom': '20px'},
                         style_cell={'textAlign': 'center','fontFamily': 'Arial'},
                         style_header={'fontWeight': 'bold'}),

    html.Button("Save Weights", id="save_the_table",n_clicks=0, style={"marginTop": "50px", "display": "block", "margin": "auto"}),
    html.Div(id="save_table_text", style={"textAlign": "center", "marginTop": "20px","color": "green"})
],style={'marginLeft': 100, 'marginRight': 100})

def register_callbacks(app):
    # PREFERENCE SETTING CALLBACKS
    @app.callback(
        Output('preference_graph', 'data'),
        Output('manual_obj', 'value'),
        Output('manual_pref', 'value'),
        Output('end_point_1_pref', 'value'),
        Output('end_point_2_pref', 'value'),
        Output('first_run','data'),
        Output('save_output', 'children'),
        Output('reset_output', 'children'),
        Output('load_output', 'children'),
        Output('total_preference', 'data'),
        Output('load_dropdown', 'options'),
        Input('dropdown_objective', 'value'),
        Input('set_obj_button', 'n_clicks'),
        Input('dropdown_decision_maker', 'value'),
        Input('set_end_button', 'n_clicks'),
        Input('save_button', 'n_clicks'),
        Input('reset_button', 'n_clicks'),
        Input('load_button', 'n_clicks'),
        State('current-page', 'data'),State('manual_obj', 'value'),
        State('manual_pref', 'value'),State('end_point_1_pref', 'value'),
        State('end_point_2_pref', 'value'),
        State('preference_graph','data'),
        State('first_run','data'),
        State('total_preference','data'),State('load_dropdown', 'value'))

    def update_graph(selected_objective, n_clicks,
                     dropdown_decision, end_n_clicks,n_clicks_save,n_clicks_reset,n_clicks_load,
                     current_page,manual_obj,manual_pref,end1_pref,end2_pref,figure,first_run,total_preference,file_name):

        #print(total_preference)
        #print(first_run)
        if current_page != "/preference":
            raise PreventUpdate

        #manual_obj = dash.callback_context.states.get("manual_obj.value")
        #manual_pref = dash.callback_context.states.get("manual_pref.value")
        #end1_pref = dash.callback_context.states.get("end_point_1_pref.value")
        #end2_pref = dash.callback_context.states.get("end_point_2_pref.value")
        # Get the min and max points based on the selected objective

        min_point, max_point = obj_overview.loc[selected_objective]['Min'], obj_overview.loc[selected_objective]['Max']

        fig = go.Figure(figure)
        fig.update_xaxes(tickformat=",")

        save_codes = weight_save_name()
        npy_files = glob.glob("*.npy")
        npy_adj = []
        for n in npy_files:
            if save_codes[dropdown_decision] in n:
                npy_adj.append(n)

        if n_clicks_save > 0 and ctx.triggered_id == 'save_button':

            x_values = list(fig.data[0]['x'])
            y_values = list(fig.data[0]['y'])
            total_preference[dropdown_decision][selected_objective] = [x_values, y_values]
            return fig.to_dict(), None, None, None, None,first_run,f'Preference saved! {total_preference[dropdown_decision][selected_objective]}', '', '', total_preference,npy_adj

        elif n_clicks_reset > 0 and ctx.triggered_id == 'reset_button':
            total_preference[dropdown_decision][selected_objective] = None

        elif n_clicks_load > 0 and file_name is not None and ctx.triggered_id == 'load_button':

            loaded_preference = np.load(file_name, allow_pickle=True).item()

            if list(loaded_preference.keys())[0] == decision_makers[0]:
                total_preference = loaded_preference
            else:
                total_preference[dropdown_decision] = loaded_preference


        if type(first_run) == int:
            first_run = [0, dropdown_decision, selected_objective]

        if dropdown_decision != first_run[1] or selected_objective != first_run[2]:
            first_run = [0, dropdown_decision, selected_objective]

        if total_preference[dropdown_decision][selected_objective] != None:

            fig.update_layout(
                title=f'{selected_objective} Preference')

            fig.data[0]['x'] = np.array(total_preference[dropdown_decision][selected_objective][0])
            fig.data[0]['y'] = np.array(total_preference[dropdown_decision][selected_objective][1])
            fig.update_layout(
                title=f'Preference Curve',
                xaxis_title=f'{selected_objective}{objective_units[selected_objective]}',
                yaxis_title='Preference Score',
                showlegend=True)

        elif first_run[0] == 0:

            # Create a plotly graph
            fig = go.Figure(layout_yaxis_range=[-100, 100])
            fig.update_xaxes(tickformat=",")

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
                title=f'Preference Curve',
                xaxis_title=f'{selected_objective}{objective_units[selected_objective]}',
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

            return fig.to_dict(), None, None, None, None,first_run,'','','',total_preference,npy_adj

        if end_n_clicks > 0 and end1_pref is not None and end2_pref is not None:
            new_end_obj, new_end_pref = [min_point, max_point], [end1_pref, end2_pref]

            fig.data[0]['x'] = np.array(new_end_obj)
            fig.data[0]['y'] = np.array(new_end_pref)

            return fig.to_dict(), None, None, None, None,first_run,'','','',total_preference,npy_adj

        return fig.to_dict(), None, None, None, None,first_run,'','','',total_preference,npy_adj

    @app.callback(
        Output('write_output', 'children'),
        Input('write_button', 'n_clicks'),
        State('write_input', 'value'),State('total_preference','data'),State('dropdown_decision_maker', 'value'))

    def save_file(n_clicks_wr, file_name,total_preference,decision_maker):
        '''
        if current_page != "/preference":
            raise PreventUpdate  # Prevent callback if we're not on the "Choice" page
        '''

        if n_clicks_wr > 0 and ctx.triggered_id == 'write_button' :
            all_save_names = weight_save_name()
            np.save(all_save_names[decision_maker] + '_' + file_name + '.npy', total_preference[decision_maker])

            return f'Saved all preference curves as {all_save_names[decision_maker] + "_" + file_name}.npy'
        else:
            # Return default values when the button hasn't been clicked
            return ""  # Empty string for 'write_output.children', current npy_files for 'load_dropdown.options'

    @app.callback(
        Output('preference_plot', 'figure'),
        Input('preference_graph', 'data'))

    def make_the_graph(figure):
        fig = go.Figure(figure)
        return fig

    @app.callback(
        Output('save_table_text','children'),
        Input('save_the_table','n_clicks'),
        State('table_app1','data'),State('dropdown_decision_maker', 'value'))

    def record_dm_weights(n_clicks_svt, table_weights, decision_maker):

        if n_clicks_svt > 0 and ctx.triggered_id == 'save_the_table':
            #print(table_weights)
            for a in table_weights:
                for b in a:

                    #print(a[b])
                    try:
                        # Convert to float (or int if needed)
                        a[b] = float(a[b])
                    except (ValueError, TypeError):
                        a[b] = a[b]

            all_save_names = weight_save_name()
            this_save_name = all_save_names[decision_maker]
            with open(this_save_name + '.json', "w") as f:
                json.dump(table_weights, f)
            return 'Weights Saved'
        return ''


    @app.callback(
        Output('table_app1','data'),
        Input('dropdown_decision_maker', 'value'))

    def default_table(decision_maker):

        all_save_names = weight_save_name()
        this_save_name = all_save_names[decision_maker]

        if os.path.exists(this_save_name + '.json'):
            with open(this_save_name + '.json', "r") as file:
                return json.load(file)
        else:
            basic_table_data[0].update({'Decision-Maker': decision_maker})
            return basic_table_data
