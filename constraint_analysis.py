# %%
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 7)  # makes figures larger by default

# %%
def create_functions():
    """
    This function creates functions needed for the constraint analysis and returns
    them as a dictionary
    """
    
    def T_W_to_P_W(T_W, v, eta_p, eta_m):
        return v*T_W/(eta_p*eta_m)

    AR = 6
    e = 0.7
    Cd0 = 0.05
    rho = 1.225
    eta_p = 0.7
    eta_m = 0.8
    k = 1/(e*AR*np.pi)
    g = 9.8
    V_stall = 6.9
    s_to = 2 # takeoff length
    CL_max = 1.34
    V_ss = 15 # m/s

    def steady_P_W(W_S):
        v = V_ss
        q = 0.5*rho*v**2
        T_W_ss = q*Cd0/W_S + k*W_S/q
        return T_W_to_P_W(T_W_ss, v, eta_p, eta_m)

    def turn_P_W(W_S):
        v = 8  # turn at 6 m/s
        q = 0.5*rho*v**2
        bank_angle = np.deg2rad(45)
        n = 1/np.cos(bank_angle) # load factor
        T_W_turn = q*Cd0/W_S + n**2*k*W_S/q
        return T_W_to_P_W(T_W_turn, v, eta_p, eta_m)

    def takeoff_P_W(W_S):
        v = V_stall*1.2 # takeoff velocity
        mu = 0.1  # rolling friction coefficient
        T_W_to = 1.2**2/(rho*g*s_to*CL_max)*W_S \
            + 0.7*Cd0/CL_max + mu
        return T_W_to_P_W(T_W_to, v, eta_p, eta_m)

    def ceiling_P_W(W_S):
        Cl = np.sqrt(3*Cd0/k)
        V_ceil = np.sqrt(2*W_S/(rho*Cl))
        return V_ceil/(0.866*eta_m*eta_p*g)

    def land_P_W(W_S):
        v = V_stall*1.2 # takeoff velocity
        mu = 0.1  # rolling friction coefficient
        T_W_to = 1.2**2/(rho*g*s_to*CL_max*1.5)*W_S \
            + 0.7*Cd0/CL_max + mu
        return T_W_to_P_W(T_W_to, v, eta_p, eta_m)

    def land_dist(W_S):
        mu = 0.1
        beta = 1
        return W_S*1.69*beta/(rho*g*mu*CL_max)
    
    return {
        'turn_P_W': turn_P_W,
        'takeoff_P_W': takeoff_P_W,
        'ceiling_P_W': ceiling_P_W,
        'land_dist': land_dist,
        'land_P_W': land_P_W,
        'steady_P_W': steady_P_W,
    }

def solve(funcs):
    """
    Takes a dictionary of functions and then solves the constraint analysis problem
    """
    
    # declare symbolic variables for the design vector we are solving for
    W_S = ca.SX.sym('W_S') # wing loading, weight/(wing area)
    P_W = ca.SX.sym('P_W') # power loading, power/weight
    # L_dist = ca.SX.sym('L_dist')# landing dist
    solver = ca.nlpsol(
        'problem',  # name
        'ipopt',  # solver
        {  # problem details
            'x': ca.vertcat(W_S, P_W),  # decision variables/ design vector
            'f': 1*P_W**2, #+ 10*L_dist**2,  # objective function
            'g': ca.vertcat(
                P_W - funcs['turn_P_W'](W_S),  # we subtract from P_W here to make g=0 when constraint is satisfied
                P_W - funcs['takeoff_P_W'](W_S),
                P_W - funcs['ceiling_P_W'](W_S),
                P_W - funcs['land_P_W'](W_S),
                P_W - funcs['steady_P_W'](W_S),
            )
        },
        {  # solver options
            'print_time': 0,
            'ipopt': {
                'sb': 'yes',
                'print_level': 0,
            }
        }
    )

    # Solve the problem
    res = solver(
        x0=[0.5, 0.5], # initial guess for (W_S, P_W)
        lbg=[0, 0, 0, 0, 0],  # lower bound on constraints, 0, must meet power loading requirement
        ubg=[ca.inf, ca.inf, ca.inf, ca.inf, ca.inf],  # upper bound on constraints, none, can have excess power loading, inf
        lbx=[0, 0],  # lower bound on state, 0, cannot have negative wing loading, power loading
        ubx=[ca.inf, ca.inf],  # upper bound on state, inf
    )
    stats = solver.stats()
    
    # If the solver failed, raise an exception
    if not stats['success']:
        raise RuntimeError(stats['return_status'])
    return {
        'W_S': float(res['x'][0]),
        'P_W': float(res['x'][1]),
    }

def constraint_analysis():
    """
    The main function that creates functions, solves the optimization problem, and then plots the constraint
    analysis diagram with the optimal design point labelled.
    """
    funcs = create_functions()
    opt_sol = solve(funcs)
    W_S_val = np.arange(5, 30, 0.1)
    plt.figure()
    plt.plot(W_S_val, funcs['turn_P_W'](W_S_val), label='turn')
    plt.plot(W_S_val, funcs['takeoff_P_W'](W_S_val), label='takeoff')
    plt.plot(W_S_val, funcs['ceiling_P_W'](W_S_val), label='ceiling')
    plt.plot(W_S_val, funcs['land_P_W'](W_S_val), label='landing')
    plt.plot(W_S_val, funcs['steady_P_W'](W_S_val), label='steady flight')
    plt.plot(opt_sol['W_S'], opt_sol['P_W'], 'r.', markersize=20, label='design point')
    plt.text(opt_sol['W_S'], opt_sol['P_W'] + 0.1, 'W_S: {W_S:0.2f}\nP_W: {P_W:0.2f}'.format(**opt_sol))
    plt.xlabel('Wing Loading, N/m^2')
    plt.ylabel('Power to Weight Ratio, Watts/N')
    plt.grid()
    plt.legend()
    plt.title('Constraint Analysis')
    plt.show()
    
constraint_analysis()

# %%


# %%
