
import pandas as pd
from scipy.optimize import minimize

from ObjFun import *


def obj_names():
    objective_names = ['NPV', 'Noise - Oss', 'Noise - Den Bosch', 'Bird Mortality',
                       'Particle Pollution', 'Energy - Oss', 'Energy - Den Bosch', 'Project Time']
    return objective_names
def dec_mak():
    decision_makers = ['Energy Provider', 'Local Residents - Oss', 'Local Residents - Den Bosch', 'Ecologists',
                       'RIVM', 'Oss Municipality', 'Den Bosch Municipality']
    return decision_makers

def bounds_cons():
    b1 = (0.1, 3.5)  # Distance to city centre (Oss) km
    b2 = (0.1, 5)  # Distance to Den Bosch km
    b3 = (1, 12)  # Number of turbines Oss
    b4 = (1, 20)  # Number of turbines Den Bosch
    b5 = (50, 150)  # Turbine hub height Oss
    b6 = (50, 150)  # Turbine hub height Den Bosch

    bounds = (b1, b2, b3, b4, b5, b6)

    cons = []

    return bounds,cons

def get_SODO_stuff():

    bounds,cons = bounds_cons()

    objective_names = obj_names()

    decision_makers = dec_mak()

    # due to the non-linearity, it might be needed to increase the maximum number of iterations for minimize

    result1 = minimize(obj_NPV, x0=np.array([0.1, 0.1, 12, 20, 150, 150]), bounds=bounds,
                   constraints=[], options={'maxiter': 1000})
    result2 = minimize(obj_noise_disturbance_oss, x0=np.array([3.5, 5, 1, 1, 50, 50]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result3 = minimize(obj_noise_disturbance_bosch, x0=np.array([3.5, 5, 1, 1, 50, 50]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result4 = minimize(obj_bird_mortality, x0=np.array([2, 2, 1, 1, 50, 50]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result5= minimize(obj_particle_pollution, x0=np.array([2, 2, 1, 1, 50, 50]), bounds=bounds,
                   constraints=[], options={'maxiter': 1000})
    result6 = minimize(obj_energy_oss, x0=np.array([0.1, 0.1, 12, 20, 150, 150]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result7 = minimize(obj_energy_bosch, x0=np.array([0.1, 0.1, 12, 20, 150, 150]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result8 = minimize(obj_project_time, x0=np.array([2, 2, 1, 1, 50, 50]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})

    result9 = minimize(obj_NPV_max, x0=np.array([3.5, 5, 1, 1, 50, 50]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result10 = minimize(obj_noise_disturbance_max_oss, x0=np.array([0.1, 0.1, 12, 20, 150, 150]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result11 = minimize(obj_noise_disturbance_max_bosch, x0=np.array([0.1, 0.1, 12, 20, 150, 150]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result12 = minimize(obj_bird_mortality_max, x0=np.array([2, 4, 12, 20, 150, 150]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result13 = minimize(obj_particle_pollution_max, x0=np.array([2, 2, 12, 20, 150, 150]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result14 = minimize(obj_energy_oss_max, x0=np.array([3.5, 5, 1, 1, 50, 50]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result15 = minimize(obj_energy_bosch_max, x0=np.array([3.5, 5, 1, 1, 50, 50]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})
    result16 = minimize(obj_project_time_max, x0=np.array([2, 2, 12, 20, 150, 150]), bounds=bounds,
                   constraints=[],options={'maxiter': 1000})

    min_obj = [result9.fun,result2.fun,result3.fun,result4.fun,result5.fun,result14.fun,result14.fun,result8.fun]
    max_obj = [-result1.fun,-result10.fun,-result11.fun,-result12.fun,-result13.fun,-result6.fun,-result7.fun,-result16.fun]

    obj_overview_dict = {'Min': min_obj,'Max': max_obj}

    obj_overview = pd.DataFrame(obj_overview_dict,index = objective_names)

    total_preference = {
        decision_maker: {objective: None for objective in objective_names}
        for decision_maker in decision_makers
    }
    return obj_overview, total_preference


