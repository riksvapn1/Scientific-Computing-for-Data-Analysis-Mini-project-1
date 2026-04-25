import numpy as np 
import matplotlib.pyplot as plt

# ==================Task 1 implementation===================

# --Parameters ---
N = 0 
T = 60 
h = 0.1 
D = 0.02 
u = np.array([0.3, 0])  
eps = 0.1 
num_steps = int(T/h)
Q = 100           # new particles per second location 0,0 
particles = np.zeros((int(Q*T), 2)) # total array for total particles
plot_idx = 0


# ----Simulation-----
                                                                                                                                                                                          
#-------fig setup---------
fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
axes = axes.flatten() 


for step in range(1,num_steps+1): 
    
    
    Z = np.random.standard_normal((N,2))

    # Euler–Maruyama step: 
    particles[0: N, :] += u*h + np.sqrt(2*D*h)*Z
    N += int(Q*h) # new amount of particles per step

    if step in[150,300,450,600]:
        t = step *h
        ax = axes[plot_idx]
        ax.scatter(particles[0:N,0], particles[0:N, 1], s=1, alpha=0.6, color='blue')
        ax.set_title(f"Time: {t}s")
        ax.set_xlim(0, 25)  
        ax.set_ylim(-5, 5)  
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plot_idx +=1

plt.tight_layout()
plt.show()

# =============================================Task 2 implementation====================================

# same setup as for task one
particles = np.zeros((int(Q*T), 2)) # total array for total particles
N = 0 
total_particles_at_end = int(Q * T) # Constant normalization factor

# ---Grid Setup ---
x_range = np.linspace(-1, 20, 200) # Matched to Task 1 plot limits
y_range = np.linspace(-3.5, 3.5, 100) # Matched to Task 1 plot limits
X_grid, Y_grid = np.meshgrid(x_range, y_range) 

# Adjusted figure size to be identical to Task 1
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.flatten() 
plot_idx = 0
norm_factor = 1 / (2 * np.pi * eps**2)

# Snapshots to match Task 1 (15s, 30s, 45s, 60s)
output_times = [15, 30, 45, 60]
output_steps = [int(t / h) for t in output_times]

# Pre-calculate to find global maximum concentration for consistent scale
all_C = []
temp_particles = np.zeros((int(Q*T), 2))
temp_N = 0
for step in range(1, num_steps + 1):
    Z = np.random.standard_normal((temp_N, 2))
    temp_particles[0: temp_N, :] += u*h + np.sqrt(2*D*h)*Z
    temp_N += int(Q*h)
    if step in output_steps:
        C = np.zeros_like(X_grid)
        for k in range(temp_N):
            xp, yp = temp_particles[k, 0], temp_particles[k, 1]
            dist_sq = (X_grid- xp)**2 + (Y_grid- yp)**2
            C += norm_factor * np.exp(-dist_sq / (2 * eps**2))
        C /= total_particles_at_end
        all_C.append(C)
v_max_global = max(C.max() for C in all_C)

# ---  sim loop ---
plot_idx = 0
N = 0
particles = np.zeros((int(Q*T), 2))
for step in range(1, num_steps + 1): 
    Z = np.random.standard_normal((N,2))
    # Euler–Maruyama step: 
    particles[0: N, :] += u*h + np.sqrt(2*D*h)*Z
    N += int(Q*h) # new amount of particles per step
    
    if step in output_steps:
        C = all_C[plot_idx]
        
        # Plot on the current subplot
        ax = axes[plot_idx]
       
        cp = ax.contourf(X_grid, Y_grid, C, levels=100, cmap='hot_r', vmin=0, vmax=v_max_global)
        
        ax.set_title(f"Concentration at t = {step*h:.1f}s")
        ax.set_xlabel("x [meters]")
        ax.set_ylabel("y [meters]")
        ax.set_xlim(-1, 20)
        ax.set_ylim(-3.5, 3.5)
        plot_idx += 1

# Shared colorbar management
fig.subplots_adjust(bottom=0.2, hspace=0.3)
cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
fig.colorbar(cp, cax=cbar_ax, orientation='horizontal', label="Concentration $C(x,t)$")
plt.show()