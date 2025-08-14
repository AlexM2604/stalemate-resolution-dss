import numpy as np
import math

#CONSTANTS
rho = 1.224 # kg/m^3 - air density
e = 0.3 # efficiency
EP = 0.06 # price of electricity per kWh
L_w = 104 # dB - example sound power level of a turbine
wind_base_speed = 3 # m/s
height_base = 50 # m
C = 0.005 # Bird collision factor 0.5%
r = 1 # rain impact
f = 0.001 # pollution probability factor 0.1%
cap_fact = 0.3 # Capacity Factor for onshore wind farms (2023)

IRR = 4.5 #%

# Calculate wind farm project NPV

def calculate_npv_l(park_power_o,park_power_b,interest_r,profit_year,hub_height_o,hub_height_b):
    
    CAPEX = 1_600_000 # EUR per MW
    O_and_M = 30722 #EUR per MW per year
    park_power = park_power_o + park_power_b
    
    flows = [0] * 31
    discounted_flows = [0] * 31
    coeff = [0] * 31
    
    for i in enumerate(coeff):
        coeff[i[0]] = (1 + (IRR/100))**i[0]
        
    for j in enumerate(flows):
        if j[0] == 4:
            flows[j[0]] = -(CAPEX * park_power)
        elif j[0] >= 6:
            flows[j[0]] = (profit_year * cap_fact) - (O_and_M * park_power)

    flows[-1] = -((0.1 * hub_height_o + 31) * park_power_o * 1000) - ((0.1 * hub_height_b + 31) * park_power_b * 1000) #End of life costs

    for k in enumerate(discounted_flows):
        discounted_flows[k[0]] = (flows[k[0]])/(coeff[k[0]])
    if park_power >= 80 and park_power <= 300:
        extension = [0] * 3
        discounted_flows = extension + discounted_flows
    elif park_power > 300:
        extension = [0] * 6
        discounted_flows = extension + discounted_flows

    NPV = sum(discounted_flows)
    
    return NPV

# Objectives:

def obj_NPV(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    global wind_oss
    global wind_bosch
    
    if x5 >= 50 and x5 <= 75:
        wind_oss = 6.97
    elif x5 > 75 and x5 <= 125:
        wind_oss = 8.16
    elif x5 > 125:
        wind_oss = 9.3

    if x6 >= 50 and x6 <= 75:
        wind_bosch = 6.97
    elif x6 > 75 and x6 <= 125:
        wind_bosch = 8.16
    elif x6 > 125:
        wind_bosch = 9.3

    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    P_oss = 0.5 * A_oss * rho * e * (wind_oss**3) # Power capacity per turbine (W) - Oss
    P_dt_oss = P_oss * 24/1000 # Converting to kWh per day
    R_oss = (x5 + wind_oss)/12 # Resistance factor for energy efficiency calculation
    profit_d_oss = (P_dt_oss * EP * x3) - (R_oss*x1) #daily

    farm_capacity_oss = (P_oss * x3) / 1_000_000 # in MW

    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    P_bosch = 0.5 * A_bosch * rho * e * (wind_bosch**3) # Power capacity per turbine (W) - Den Bosch
    P_dt_bosch = P_bosch * 24/1000 # Converting to kWh per day
    R_bosch = (x6 + wind_bosch)/12 # Resistance factor for energy efficiency calculation
    profit_d_bosch = (P_dt_bosch * EP * x4) - (R_bosch*x2) #daily

    farm_capacity_bosch = (P_bosch * x4) / 1_000_000 # in MW

    farm_capacity_all = farm_capacity_oss + farm_capacity_bosch

    profit_year_all = (profit_d_oss * 365) + (profit_d_bosch * 365)

    NPV = calculate_npv_l(farm_capacity_oss,farm_capacity_bosch,IRR,profit_year_all,x5,x6)

    #print(f'The wind farm capacity is {farm_capacity} MW')

    return -NPV

def obj_NPV_max(variables):

    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    global wind_oss
    global wind_bosch
    
    if x5 >= 50 and x5 <= 75:
        wind_oss = 6.97
    elif x5 > 75 and x5 <= 125:
        wind_oss = 8.16
    elif x5 > 125:
        wind_oss = 9.3

    if x6 >= 50 and x6 <= 75:
        wind_bosch = 6.97
    elif x6 > 75 and x6 <= 125:
        wind_bosch = 8.16
    elif x6 > 125:
        wind_bosch = 9.3

    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    P_oss = 0.5 * A_oss * rho * e * (wind_oss**3) # Power capacity per turbine (W) - Oss
    P_dt_oss = P_oss * 24/1000 # Converting to kWh per day
    R_oss = (x5 + wind_oss)/12 # Resistance factor for energy efficiency calculation
    profit_d_oss = (P_dt_oss * EP * x3) - (R_oss*x1) #daily

    farm_capacity_oss = (P_oss * x3) / 1_000_000 # in MW

    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    P_bosch = 0.5 * A_bosch * rho * e * (wind_bosch**3) # Power capacity per turbine (W) - Den Bosch
    P_dt_bosch = P_bosch * 24/1000 # Converting to kWh per day
    R_bosch = (x6 + wind_bosch)/12 # Resistance factor for energy efficiency calculation
    profit_d_bosch = (P_dt_bosch * EP * x4) - (R_bosch*x2) #daily

    farm_capacity_bosch = (P_bosch * x4) / 1_000_000 # in MW

    farm_capacity_all = farm_capacity_oss + farm_capacity_bosch

    profit_year_all = (profit_d_oss * 365) + (profit_d_bosch * 365)

    NPV = calculate_npv_l(farm_capacity_oss,farm_capacity_bosch,IRR,profit_year_all,x5,x6)

    #print(f'The wind farm capacity is {farm_capacity} MW')

    return NPV

def obj_noise_disturbance_oss(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch

    if x5 >= 50 and x5 <= 75:
        wind_oss = 6.97
    elif x5 > 75 and x5 <= 125:
        wind_oss = 8.16
    elif x5 > 125:
        wind_oss = 9.3
    
    wind_adj = (wind_oss - wind_base_speed) * 0.2 # Increase in noise with the wind
    height_adj = (x5 - height_base) * 0.1 # Increase in noise with height
    L_d_single = L_w + wind_adj + height_adj - (20 * np.log10((x1*1000)/1)) #Noise
                                                        # level at a distance d
    SP = (pow(10, (L_d_single/10))) # Convert single turbine noise from dB to power
    
    noise = 10 * (np.log10(x3 * SP))
    
    return noise  

def obj_noise_disturbance_bosch(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    if x6 >= 50 and x6 <= 75:
        wind_bosch = 6.97
    elif x6 > 75 and x6 <= 125:
        wind_bosch = 8.16
    elif x6 > 125:
        wind_bosch = 9.3
    
    wind_adj = (wind_bosch - wind_base_speed) * 0.2 # Increase in noise with the wind
    height_adj = (x6 - height_base) * 0.1 # Increase in noise with height
    L_d_single = L_w + wind_adj + height_adj - (20 * np.log10((x2*1000)/1)) #Noise
                                                        # level at a distance d
    SP = (pow(10, (L_d_single/10))) # Convert single turbine noise from dB to power
    
    noise = 10 * (np.log10(x4 * SP))
    
    return noise 

def obj_bird_mortality(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    split = 200
    
    #print(type(x1))
    
    if type(x1) == np.float64 or type(x1) == float:
        if x2 > 3 and x2 < 5:
            split = 800
        else: 
            split = 200
    else:
        split = 'idk yet'
    #[workvol_foundation[i] for i in x1]
    
    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    M_oss = (A_oss/30000) * C * split * x3 # bird mortality per turbine
    M_bosch = (A_bosch/30000) * C * split * x4 # bird mortality per turbine
    #have to come up with a way to set split = 800 if 8<x1<9 and 200 in other cases
    mortality = M_oss + M_bosch
    
    return mortality

def obj_particle_pollution(variables):

    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    

    if x5 >= 50 and x5 <= 75:
        wind_oss = 6.97
    elif x5 > 75 and x5 <= 125:
        wind_oss = 8.16
    elif x5 > 125:
        wind_oss = 9.3

    if x6 >= 50 and x6 <= 75:
        wind_bosch = 6.97
    elif x6 > 75 and x6 <= 125:
        wind_bosch = 8.16
    elif x6 > 125:
        wind_bosch = 9.3
    
    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    
    w_oss = wind_oss**3
    w_bosch = wind_bosch**3
    E_oss = w_oss * f * r * (A_oss/30000)
    E_bosch = w_bosch * f * r * (A_bosch/30000)
    
    pollution = (E_oss * x3) + (E_bosch * x4)
    return pollution

def obj_energy_oss(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    global wind_oss
    
    if x5 >= 50 and x5 <= 75:
        wind_oss = 6.97
    elif x5 > 75 and x5 <= 125:
        wind_oss = 8.16
    elif x5 > 125:
        wind_oss = 9.3

    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    P_oss = 0.5 * A_oss * rho * e * (wind_oss**3) # Power capacity per turbine (W) - Oss
    P_dt_oss = P_oss * 24/1000 # Converting to kWh per day
    R_oss = (x5 + wind_oss)/12 # Resistance factor for energy efficiency calculation
    profit_d_oss = (P_dt_oss * EP * x3) - (R_oss*x1) #daily

    farm_capacity_oss = (P_oss * x3) / 1_000_000 # in MW

    return -farm_capacity_oss

def obj_energy_bosch(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    global wind_bosch
    

    if x6 >= 50 and x6 <= 75:
        wind_bosch = 6.97
    elif x6 > 75 and x6 <= 125:
        wind_bosch = 8.16
    elif x6 > 125:
        wind_bosch = 9.3

    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    P_bosch = 0.5 * A_bosch * rho * e * (wind_bosch**3) # Power capacity per turbine (W) - Den Bosch
    P_dt_bosch = P_bosch * 24/1000 # Converting to kWh per day
    R_bosch = (x6 + wind_bosch)/12 # Resistance factor for energy efficiency calculation
    profit_d_bosch = (P_dt_bosch * EP * x4) - (R_bosch*x2) #daily

    farm_capacity_bosch = (P_bosch * x4) / 1_000_000 # in MW

    return -farm_capacity_bosch
    
def obj_noise_disturbance_max_oss(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch

    if x5 >= 50 and x5 <= 75:
        wind_oss = 6.97
    elif x5 > 75 and x5 <= 125:
        wind_oss = 8.16
    elif x5 > 125:
        wind_oss = 9.3
    
    wind_adj = (wind_oss - wind_base_speed) * 0.2 # Increase in noise with the wind
    height_adj = (x5 - height_base) * 0.1 # Increase in noise with height
    L_d_single = L_w + wind_adj + height_adj - (20 * np.log10((x1*1000)/1)) #Noise
                                                        # level at a distance d
    SP = (pow(10, (L_d_single/10))) # Convert single turbine noise from dB to power
    
    noise = 10 * (np.log10(x3 * SP))
    
    return -noise 

def obj_noise_disturbance_max_bosch(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch

    if x6 >= 50 and x6 <= 75:
        wind_bosch = 6.97
    elif x6 > 75 and x6 <= 125:
        wind_bosch = 8.16
    elif x6 > 125:
        wind_bosch = 9.3
    
    wind_adj = (wind_bosch - wind_base_speed) * 0.2 # Increase in noise with the wind
    height_adj = (x6 - height_base) * 0.1 # Increase in noise with height
    L_d_single = L_w + wind_adj + height_adj - (20 * np.log10((x2*1000)/1)) #Noise
                                                        # level at a distance d
    SP = (pow(10, (L_d_single/10))) # Convert single turbine noise from dB to power
    
    noise = 10 * (np.log10(x4 * SP))
    
    return -noise 

def obj_bird_mortality_max(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    split_oss = 200
    split_bosch = 200
    
    #print(type(x1))
    
    if type(x1) == np.float64 or type(x1) == float:
        if x2 > 3 and x2 < 5:
            split_bosch = 800
        else: 
            split_bosch = 200
    else:
        split = 'idk yet'
    #[workvol_foundation[i] for i in x1]
    
    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    M_oss = (A_oss/30000) * C * split_oss * x3 # bird mortality per turbine
    M_bosch = (A_bosch/30000) * C * split_bosch * x4 # bird mortality per turbine
    #have to come up with a way to set split = 800 if 8<x1<9 and 200 in other cases
    mortality = M_oss + M_bosch
    
    return -mortality
    
def obj_particle_pollution_max(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    

    if x5 >= 50 and x5 <= 75:
        wind_oss = 6.97
    elif x5 > 75 and x5 <= 125:
        wind_oss = 8.16
    elif x5 > 125:
        wind_oss = 9.3

    if x6 >= 50 and x6 <= 75:
        wind_bosch = 6.97
    elif x6 > 75 and x6 <= 125:
        wind_bosch = 8.16
    elif x6 > 125:
        wind_bosch = 9.3
    
    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    
    w_oss = wind_oss**3
    w_bosch = wind_bosch**3
    E_oss = w_oss * f * r * (A_oss/30000)
    E_bosch = w_bosch * f * r * (A_bosch/30000)
    
    pollution = (E_oss * x3) + (E_bosch * x4)
    return -pollution

def obj_energy_oss_max(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    global wind_oss
    
    if x5 >= 50 and x5 <= 75:
        wind_oss = 6.97
    elif x5 > 75 and x5 <= 125:
        wind_oss = 8.16
    elif x5 > 125:
        wind_oss = 9.3

    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    P_oss = 0.5 * A_oss * rho * e * (wind_oss**3) # Power capacity per turbine (W) - Oss
    P_dt_oss = P_oss * 24/1000 # Converting to kWh per day
    R_oss = (x5 + wind_oss)/12 # Resistance factor for energy efficiency calculation
    profit_d_oss = (P_dt_oss * EP * x3) - (R_oss*x1) #daily

    farm_capacity_oss = (P_oss * x3) / 1_000_000 # in MW

    return farm_capacity_oss

def obj_energy_bosch_max(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    global wind_bosch
    

    if x6 >= 50 and x6 <= 75:
        wind_bosch = 6.97
    elif x6 > 75 and x6 <= 125:
        wind_bosch = 8.16
    elif x6 > 125:
        wind_bosch = 9.3

    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    P_bosch = 0.5 * A_bosch * rho * e * (wind_bosch**3) # Power capacity per turbine (W) - Den Bosch
    P_dt_bosch = P_bosch * 24/1000 # Converting to kWh per day
    R_bosch = (x6 + wind_bosch)/12 # Resistance factor for energy efficiency calculation
    profit_d_bosch = (P_dt_bosch * EP * x4) - (R_bosch*x2) #daily

    farm_capacity_bosch = (P_bosch * x4) / 1_000_000 # in MW

    return farm_capacity_bosch

def obj_project_time(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    global wind_oss
    global wind_bosch
    
    if x5 >= 50 and x5 <= 75:
        wind_oss = 6.97
    elif x5 > 75 and x5 <= 125:
        wind_oss = 8.16
    elif x5 > 125:
        wind_oss = 9.3

    if x6 >= 50 and x6 <= 75:
        wind_bosch = 6.97
    elif x6 > 75 and x6 <= 125:
        wind_bosch = 8.16
    elif x6 > 125:
        wind_bosch = 9.3

    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    P_oss = 0.5 * A_oss * rho * e * (wind_oss**3) # Power capacity per turbine (W) - Oss
    P_dt_oss = P_oss * 24/1000 # Converting to kWh per day
    farm_capacity_oss = (P_oss * x3) / 1_000_000 # in MW

    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    P_bosch = 0.5 * A_bosch * rho * e * (wind_bosch**3) # Power capacity per turbine (W) - Den Bosch
    P_dt_bosch = P_bosch * 24/1000 # Converting to kWh per day

    farm_capacity_bosch = (P_bosch * x4) / 1_000_000 # in MW

    farm_capacity_all = farm_capacity_oss + farm_capacity_bosch

    project_time = 5.3
    
    if farm_capacity_all >= 80 and farm_capacity_all <= 300:
        project_time = 8.3
    elif farm_capacity_all > 300:
        project_time = 11.8
    
    return project_time

def obj_project_time_max(variables):
    
    x1 = variables[0] # Distance to city centre (Oss)
    x2 = variables[1] # Distance to Den Bosch
    x3 = variables[2] # Number of turbines Oss
    x4 = variables[3] # Number of turbines Den Bosch
    x5 = variables[4] # Turbine hub height Oss
    x6 = variables[5] # Turbine hub height Den Bosch
    
    global wind_oss
    global wind_bosch
    
    if x5 >= 50 and x5 <= 75:
        wind_oss = 6.97
    elif x5 > 75 and x5 <= 125:
        wind_oss = 8.16
    elif x5 > 125:
        wind_oss = 9.3

    if x6 >= 50 and x6 <= 75:
        wind_bosch = 6.97
    elif x6 > 75 and x6 <= 125:
        wind_bosch = 8.16
    elif x6 > 125:
        wind_bosch = 9.3

    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    P_oss = 0.5 * A_oss * rho * e * (wind_oss**3) # Power capacity per turbine (W) - Oss
    P_dt_oss = P_oss * 24/1000 # Converting to kWh per day
    farm_capacity_oss = (P_oss * x3) / 1_000_000 # in MW

    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    P_bosch = 0.5 * A_bosch * rho * e * (wind_bosch**3) # Power capacity per turbine (W) - Den Bosch
    P_dt_bosch = P_bosch * 24/1000 # Converting to kWh per day

    farm_capacity_bosch = (P_bosch * x4) / 1_000_000 # in MW

    farm_capacity_all = farm_capacity_oss + farm_capacity_bosch

    project_time = 5.3
    
    if farm_capacity_all >= 80 and farm_capacity_all <= 300:
        project_time = 8.3
    elif farm_capacity_all > 300:
        project_time = 11.8
    return -project_time

# Objectives:

def mort_height_ga(inp):
    
    factor = None
    
    if inp > 3 and inp < 5:
        factor = 800
    else:
        factor = 200
        
    return factor

def wind_speed_height_ga(inp):

    wspeed = None

    if inp >= 50 and inp <= 75:
        wspeed = 6.97
    elif inp > 75 and inp <= 125:
        wspeed = 8.16
    elif inp > 125:
        wspeed = 9.3
        
    return wspeed

def project_time_cond_ga(inp):

    time = 5.3

    if inp >= 80 and inp <= 300:
        time = 8.3
    elif inp > 300:
        time = 11.8
    return time

def obj_NPV_ga(x1,x2,x3,x4,x5,x6):
    
    wind_oss = np.array([wind_speed_height_ga(i) for i in x5])
    wind_bosch = np.array([wind_speed_height_ga(i) for i in x6])
    
    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    P_oss = 0.5 * A_oss * rho * e * (wind_oss**3) # Power capacity per turbine (W) - Oss
    P_dt_oss = P_oss * 24/1000 # Converting to kWh per day
    R_oss = (x5 + wind_oss)/12 # Resistance factor for energy efficiency calculation
    profit_d_oss = (P_dt_oss * EP * x3) - (R_oss*x1) #daily

    farm_capacity_oss = (P_oss * x3) / 1_000_000 # in MW

    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    P_bosch = 0.5 * A_bosch * rho * e * (wind_bosch**3) # Power capacity per turbine (W) - Den Bosch
    P_dt_bosch = P_bosch * 24/1000 # Converting to kWh per day
    R_bosch = (x6 + wind_bosch)/12 # Resistance factor for energy efficiency calculation
    profit_d_bosch = (P_dt_bosch * EP * x4) - (R_bosch*x2) #daily

    farm_capacity_bosch = (P_bosch * x4) / 1_000_000 # in MW

    farm_capacity_all = farm_capacity_oss + farm_capacity_bosch

    profit_year_all = (profit_d_oss * 365) + (profit_d_bosch * 365)

    NPV = np.array([calculate_npv_l(farm_capacity_oss[b[0]],farm_capacity_bosch[b[0]],IRR,profit_year_all[b[0]],
                                    x5[b[0]],x6[b[0]]) for b in np.ndenumerate(x5)])

    #print(f'The wind farm capacity is {farm_capacity} MW')

    return -NPV

def obj_noise_disturbance_oss_ga(x1,x2,x3,x4,x5,x6):

    wind_oss = np.array([wind_speed_height_ga(i) for i in x5])
    
    wind_adj = (wind_oss - wind_base_speed) * 0.2 # Increase in noise with the wind
    height_adj = (x5 - height_base) * 0.1 # Increase in noise with height
    L_d_single = L_w + wind_adj + height_adj - (20 * np.log10((x1*1000)/1)) #Noise
                                                        # level at a distance d
    SP = (pow(10, (L_d_single/10))) # Convert single turbine noise from dB to power
    
    noise = 10 * (np.log10(x3 * SP))
    
    return noise  

def obj_noise_disturbance_bosch_ga(x1,x2,x3,x4,x5,x6):
    
    wind_bosch = np.array([wind_speed_height_ga(i) for i in x6])
    
    wind_adj = (wind_bosch - wind_base_speed) * 0.2 # Increase in noise with the wind
    height_adj = (x6 - height_base) * 0.1 # Increase in noise with height
    L_d_single = L_w + wind_adj + height_adj - (20 * np.log10((x2*1000)/1)) #Noise
                                                        # level at a distance d
    SP = (pow(10, (L_d_single/10))) # Convert single turbine noise from dB to power
    
    noise = 10 * (np.log10(x4 * SP))
    
    return noise 

def obj_bird_mortality_ga(x1,x2,x3,x4,x5,x6):
    
    split_oss = 200
    split_bosch = np.array([mort_height_ga(i) for i in x1])
    
    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    M_oss = (A_oss/30000) * C * split_oss * x3 # bird mortality per turbine
    M_bosch = (A_bosch/30000) * C * split_bosch * x4 # bird mortality per turbine
    #have to come up with a way to set split = 800 if 8<x1<9 and 200 in other cases
    mortality = M_oss + M_bosch
    
    return mortality

def obj_particle_pollution_ga(x1,x2,x3,x4,x5,x6):

    wind_oss = np.array([wind_speed_height_ga(i) for i in x5])
    wind_bosch = np.array([wind_speed_height_ga(i) for i in x6])
    
    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    
    w_oss = wind_oss**3
    w_bosch = wind_bosch**3
    E_oss = w_oss * f * r * (A_oss/30000)
    E_bosch = w_bosch * f * r * (A_bosch/30000)
    
    pollution = (E_oss * x3) + (E_bosch * x4)
    return pollution

def obj_energy_oss_ga(x1,x2,x3,x4,x5,x6):
    
    wind_oss = np.array([wind_speed_height_ga(i) for i in x5])

    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    P_oss = 0.5 * A_oss * rho * e * (wind_oss**3) # Power capacity per turbine (W) - Oss
    P_dt_oss = P_oss * 24/1000 # Converting to kWh per day
    R_oss = (x5 + wind_oss)/12 # Resistance factor for energy efficiency calculation
    profit_d_oss = (P_dt_oss * EP * x3) - (R_oss*x1) #daily

    farm_capacity_oss = (P_oss * x3) / 1_000_000 # in MW

    return -farm_capacity_oss

def obj_energy_bosch_ga(x1,x2,x3,x4,x5,x6):
    
    wind_bosch = np.array([wind_speed_height_ga(i) for i in x6])

    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    P_bosch = 0.5 * A_bosch * rho * e * (wind_bosch**3) # Power capacity per turbine (W) - Den Bosch
    P_dt_bosch = P_bosch * 24/1000 # Converting to kWh per day
    R_bosch = (x6 + wind_bosch)/12 # Resistance factor for energy efficiency calculation
    profit_d_bosch = (P_dt_bosch * EP * x4) - (R_bosch*x2) #daily

    farm_capacity_bosch = (P_bosch * x4) / 1_000_000 # in MW

    return -farm_capacity_bosch

def obj_project_time_ga(x1,x2,x3,x4,x5,x6):
    
    wind_oss = np.array([wind_speed_height_ga(i) for i in x5])
    wind_bosch = np.array([wind_speed_height_ga(i) for i in x6])

    D_oss = x5 * 1.3 # Blade Diameter Oss
    A_oss = (math.pi) * ((D_oss/2)**2) # Swept area of wind turbine blades Oss
    P_oss = 0.5 * A_oss * rho * e * (wind_oss**3) # Power capacity per turbine (W) - Oss
    P_dt_oss = P_oss * 24/1000 # Converting to kWh per day
    farm_capacity_oss = (P_oss * x3) / 1_000_000 # in MW

    D_bosch = x6 * 1.3 # Blade Diameter Den Bosch
    A_bosch = (math.pi) * ((D_bosch/2)**2) # Swept area of wind turbine blades Den Bosch
    P_bosch = 0.5 * A_bosch * rho * e * (wind_bosch**3) # Power capacity per turbine (W) - Den Bosch
    P_dt_bosch = P_bosch * 24/1000 # Converting to kWh per day

    farm_capacity_bosch = (P_bosch * x4) / 1_000_000 # in MW

    farm_capacity_all = farm_capacity_oss + farm_capacity_bosch

    project_time = np.array([project_time_cond_ga(i) for i in farm_capacity_all])
    
    return project_time
