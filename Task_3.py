import numpy as np 
import matplotlib.pyplot as plt

# ==================Task 3 implementation===================

# --Parameters ---
Q = 100           # new particles per second location 0,0 
T = 60 
h = 0.1 
D = 0.02 
u = np.array([0.3, 0])  
eps = 0.1 
num_steps = int(T/h)
particles = np.zeros((int(Q*T), 2)) # total array for total particles

# --- Setup for Task 3 Visuals ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
axes = axes.flatten() 
plot_idx = 0
N = 0

# ----Simulation for Particle Scatter Plot-----
for step in range(1, num_steps + 1): 
    Z = np.random.standard_normal((N, 2))
    # Euler–Maruyama step: 
    particles[0: N, :] += u*h + np.sqrt(2*D*h)*Z
    N += int(Q*h) # new amount of particles per step

    if step in [150, 300, 450, 600]:
        ax = axes[plot_idx]
        ax.scatter(particles[0:N, 0], particles[0:N, 1], s=0.5, alpha=0.4, color='blue')
        ax.set_title(f"Time: {step*h:.0f}s", fontsize=14)
        ax.set_xlim(0, 25)  
        ax.set_ylim(-5, 5)  
        ax.set_xlabel("x [m]", fontsize=12)
        ax.set_ylabel("y [m]", fontsize=12)
        plot_idx += 1

plt.tight_layout()
plt.show()

# ==================Task 3 Concentration Implementation===================

# Reset for concentration estimation
particles = np.zeros((int(Q*T), 2)) 
N = 0 
norm_factor = 1 / (2 * np.pi * eps**2)
x_range = np.linspace(0, 25, 200)
y_range = np.linspace(-5, 5, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range) 

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
axes = axes.flatten() 
plot_idx = 0

# Run simulation and plot
for step in range(1, num_steps + 1): 
    Z = np.random.standard_normal((N, 2))
    particles[0: N, :] += u*h + np.sqrt(2*D*h)*Z
    N += int(Q*h)
    
    if step in [150, 300, 450, 600]:
        C = np.zeros_like(X_grid)
        for k in range(N):
            dist_sq = (X_grid - particles[k, 0])**2 + (Y_grid - particles[k, 1])**2
            C += norm_factor * np.exp(-dist_sq / (2 * eps**2))
        
        # Plot
        ax = axes[plot_idx]
        # Using Reds as requested with fixed range for visual consistency
        cp = ax.contourf(X_grid, Y_grid, C / (Q*step*h), levels=50, cmap='Reds', vmin=0, vmax=0.05)
        
        ax.set_title(f"Concentration at t = {step*h:.0f}s", fontsize=14)
        ax.set_xlabel("x [m]", fontsize=12)
        ax.set_ylabel("y [m]", fontsize=12)
        plot_idx += 1

# Shared colorbar management
fig.subplots_adjust(bottom=0.15, wspace=0.2, hspace=0.3)
cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
fig.colorbar(cp, cax=cbar_ax, orientation='horizontal', label=r"Concentration $C(x,t)$ [units/m$^2$]")

plt.show()