import jax.numpy as jnp
import jax.random as jr
from diffrax import diffeqsolve, ControlTerm, EulerHeun, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
import diffrax as dfx
import equinox as eqx
from utils import *


def diffusive_learning(control_fun, reward_fun, y0, dt=0.05, t0=0, t1=200, epsilon=1.0, lambd=1.0, sigma=0.5, eta=1.0, key=jr.PRNGKey(0), target_fun=None):

    @eqx.filter_jit
    def drift_fun(t, y, args):
        '''
            Return deterministic values:
             1) dr̄ = ε(r-r̄) dt
             2) dθ = λ(μ-θ)
             3) dμ = η(r-r̄)(θ-μ)
             4) dr = r -> Cumulative reward
        '''

        # Variable dimension in theta and mu
        N = y.size
        M = int((N - 2) / 2)  
        rbar = y[0]           
        theta = y[1:(M+1)]     
        mu = y[(M+1):(2*M+1)]  
        gain = y[-1]       
        if target_fun is None: 
            r = reward_fun(args[0](t), theta, t)
        else:
            r = reward_fun(args[0](t), theta, args[1](t))
        return jnp.hstack([epsilon * (r - rbar),
                           lambd * (mu - theta),
                           eta * (r - rbar) * (theta - mu),
                           r])  

    # Number of parameters (M)
    N = y0.size
    M = int((N - 2) / 2) 

    # Wiener process (random noise)
    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(M,), key=key)
    
    # Apply noise to theta (2nd output variable)
    D = jnp.vstack([jnp.zeros(M), sigma * jnp.eye(M), jnp.zeros((M, M)), jnp.zeros(M)]) # Diffusion matrix (sort of mask)
    diffusion_fun = lambda t, y, args: D
    noise_func = ControlTerm(diffusion_fun, brownian_motion)
    terms = MultiTerm(ODETerm(drift_fun), noise_func)

    # Numerical integration to simulate system
    solver = EulerHeun()
    if target_fun is None:
        sol = diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt, y0=y0, saveat=dfx.SaveAt(steps=True), args=[control_fun], max_steps=100000)
    else:
        sol = diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0, saveat=dfx.SaveAt(steps=True), args=[control_fun, target_fun], max_steps=300000)

    # Remove infinite values from the solution
    ts = sol.ts[~jnp.isinf(sol.ts)]
    ys = sol.ys[~jnp.isinf(sol.ts), :]

    return ts, ys



# Updates 3 sets of parameters: (i) input to hidden state, (ii) recurrent hidden state, (iii) readout
def diffusive_learning_recurrent(net_structure, control_fun, reward_fun, y0, dt=0.05, t0=0, t1=150.0, epsilon=1.0, lambd=10.0, sigma=0.05, eta=1.0, key=jr.PRNGKey(0), target_fun=None):

    @eqx.filter_jit
    def drift_fun(t, y, args):
        '''
            Return deterministic values:
             1) dr̄ = ε(r-r̄) dt
             2) dz = tanh(θ1*z+θ2*x)
             3) dztg = tanh(θ1tg*ztg+θ2tg*x)
             4) dθ1 = λ(μ1-θ1)
             5) dμ1 = η(r-r̄)(θ1-μ1)
             6) dθ2 = λ(μ2-θ2)
             7) dμ2 = η(r-r̄)(θ2-μ2)
             8) dθ3 = λ(μ3-θ3)
             9) dμ3 = η(r-r̄)(θ3-μ3)
             10) dr = r -> Cumulative reward
        '''

        # Extract params from the input (we need the squeeze idk why)
        M = net_structure[0]
        rbar = y[0]           
        z = y[1] 
        z_gt = y[2] 
        theta1 = jnp.squeeze(y[3:(M+3)])
        mu1 = jnp.squeeze(y[(M+3):(2*M+3)])
        theta2 = jnp.squeeze(y[(2*M+3):(2*M+4)])     
        mu2 = jnp.squeeze(y[(2*M+4):(2*M+5)])  
        theta3 = jnp.squeeze(y[(2*M+5):(2*M+6)])     
        mu3 = jnp.squeeze(y[(2*M+6):(2*M+7)])  
        gain = y[-1]      
        
        
        # Deterministic part
        if target_fun is None: # Given target parameters
            control_fun = args[0]
            [theta1_gt, theta2_gt, theta3_gt] = [0.3, 0.7, 1.0]
            x = control_fun(t) # Input
            r = reward_fun(theta3 * z, theta3_gt * z_gt, t)
            return jnp.array([epsilon * (r - rbar),
                            tanh(theta1 * z + theta2 * x) - z,
                            tanh(theta1_gt * z_gt + theta2_gt * x) - z_gt,
                            lambd * (mu1 - theta1), 
                            eta * (r - rbar) * (theta1 - mu1),
                            lambd * (mu2 - theta2), 
                            eta * (r - rbar) * (theta2 - mu2),
                            lambd * (mu3 - theta3), 
                            eta * (r - rbar) * (theta3 - mu3),
                            r                        
                            ])
        else: # Given target output
            control_fun = args[0]
            x = control_fun(t) # Input
            r = reward_fun(z, theta3, args[1](t))
            return jnp.array([epsilon * (r - rbar),
                            tanh(theta1 * z + theta2 * x) - z,
                            0,
                            lambd * (mu1 - theta1), 
                            eta * (r - rbar) * (theta1 - mu1),
                            lambd * (mu2 - theta2), 
                            eta * (r - rbar) * (theta2 - mu2),
                            lambd * (mu3 - theta3), 
                            eta * (r - rbar) * (theta3 - mu3),
                            r                        
                            ])

    # # Variable dimension in theta and mu
    M = net_structure[0]

    # Wiener process (random noise)
    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(M+2,), key=key)

    # Apply noise to theta 
    D1 = jnp.vstack([jnp.zeros(M), jnp.zeros(M), jnp.zeros(M), sigma * jnp.eye(M), jnp.zeros((M, M)), jnp.zeros(M), jnp.zeros(M), jnp.zeros(M), jnp.zeros(M), jnp.zeros(M)]) 
    D2 = jnp.vstack([jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros((M, 1)), jnp.zeros((M, 1)), sigma * jnp.ones(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)]) 
    D3 = jnp.vstack([jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros((M, 1)), jnp.zeros((M, 1)), jnp.zeros(1), jnp.zeros(1), sigma * jnp.ones(1), jnp.zeros(1), jnp.zeros(1)]) 
    D = jnp.hstack([D1, D2, D3])
    diffusion_fun = lambda t, y, args: D
    terms = MultiTerm(ODETerm(drift_fun), ControlTerm(diffusion_fun, brownian_motion))

    # Numerical integration to simulate system
    solver = EulerHeun()
    if target_fun is None:
        sol = diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt, y0=y0, saveat=dfx.SaveAt(steps=True), args=[control_fun], max_steps=100000)
    else:
        sol = diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt, y0=y0, saveat=dfx.SaveAt(steps=True), args=[control_fun, target_fun], max_steps=300000)

    # Remove infinite values from the solution
    ts = sol.ts[~jnp.isinf(sol.ts)]
    ys = sol.ys[~jnp.isinf(sol.ts), :]

    return ts, ys