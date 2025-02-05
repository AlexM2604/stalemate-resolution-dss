from Calc_Sodo import *
import json
from scipy.interpolate import pchip_interpolate
from genetic_algorithm_pfm import GeneticAlgorithm
import plotly.express as px
import plotly.graph_objects as go


f = open('save_data.json')
final_weights = json.load(f)
df_not_used,final_pref = get_SODO_stuff()
objective_names = obj_names()

bounds,cons_ga = bounds_cons()

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

def preferendus_go():

    # specify the number of runs of the optimization
    n_runs = 5

    # make dictionary with parameter settings for the GA
    #print('Run IMAP')
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
    ga = GeneticAlgorithm(objective=objective, constraints=cons_ga, bounds=bounds, options=options,verbose = False)  # initialize GA

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

    return pref_final_all,res,obj_val

#I think I have to move these 2 functions to the file in which I will call the dashboard

def setup_for_display(final_pref,objective_names,obj_val,pref_final_all):
    labels = []
    for b in final_pref:
        for obj in objective_names:
            if final_pref[b][obj] != None:
                labels.append(f'{b},{obj}')

    dict_for_final_df = {'Objective Value': obj_val[0], 'Preference Score': pref_final_all[0]}
    result_df = pd.DataFrame(data=dict_for_final_df, index=labels)

    return labels,result_df


def plot_polar_ppi(pref_scores, pref_names):
    # Create DataFrame for polar plot data
    df = pd.DataFrame(dict(r=pref_scores, theta=pref_names))

    # Create the polar plot for the main data
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r=[0, 100])
    fig.update_traces(fill='toself')

    # Add circular dots for each vertex, color based on r value
    fig.add_trace(go.Scatterpolar(
        r=pref_scores,
        theta=pref_names,
        mode='markers',
        marker=dict(
            size=10,  # Size of the dots
            color=['green' if r > 0 else 'red' for r in pref_scores],  # Green if r > 0, else red
            # line=dict(width=1, color='black')  # Optional: Add a black border around the dots
        ),
        name='Score',
        showlegend=False
    ))

    return fig

