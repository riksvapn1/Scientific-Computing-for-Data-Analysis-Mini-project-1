import numpy as np
import matplotlib.pyplot as plt


# --Parameters ---
N = 2000 
T = 60 
h = 0.1 
D = 0.02 
u = np.array([0.3, 0])  
eps = 0.1 
particles = np.zeros((N, 2))
num_steps = int(T/h)

# ---Grid Setup ---
x_range = np.linspace(0, 25, 200)
y_range = np.linspace(-5, 5, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range) # X and Y_grid same size

# Set fixed concentration limits for consistent color mapping
v_min, v_max = 0, 0.25 

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
axes = axes.flatten() 
plot_idx = 0

norm_factor = 1 / (2 * np.pi * eps**2)

# ---  sim loop ---
for step in range(1, num_steps + 1): 
    
    Z = np.random.standard_normal((N, 2))
    particles += u * h + np.sqrt(2 * D * h) * Z # step
    
    if step in [150, 300, 450, 600]:
        C = np.zeros_like(X_grid)

        
        for k in range(N):
            xp, yp = particles[k, 0], particles[k, 1] # every particle coord
            
            # Distance squared (x^T x)
            dist_sq = (X_grid - xp)**2 + (Y_grid - yp)**2
            
            
            C += norm_factor * np.exp(-dist_sq / (2 * eps**2))
        
        # Average the field
        C /= N
        
        # Plot on the current subplot
        ax = axes[plot_idx]
       
        # Added vmin/vmax for visual consistency across plots
        cp = ax.contourf(X_grid, Y_grid, C, levels=50, cmap='Reds', vmin=v_min, vmax=v_max)
        
        
        ax.set_title(f"Concentration at t = {step*h:.1f}s", fontsize=12)
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel("y [m]", fontsize=10)
        # Removed individual colorbar from here to avoid squeezing
        plot_idx += 1

# Added shared colorbar at the bottom
fig.subplots_adjust(bottom=0.15, wspace=0.2, hspace=0.3)
cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
fig.colorbar(cp, cax=cbar_ax, orientation='horizontal', label="Concentration $C(x,t)$")

print(x_range)

plt.show()