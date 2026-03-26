import numpy as np
import matplotlib.pyplot as plt


# --- 1. Parameters ---
N = 2000 
T = 60 
h = 0.1 
D = 0.02 
u = np.array([0.3, 0])  
eps = 0.1 
particles = np.zeros((N, 2))
num_steps = int(T/h)

# --- 2. Grid Setup ---
x_range = np.linspace(0, 25, 200)
y_range = np.linspace(-5, 5, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

norm_factor = 1 / (2 * np.pi * eps**2)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten() # Turn 2x2 grid into a simple list of 4 spots
plot_idx = 0

# --- 4. Simulation Loop ---
for step in range(1, num_steps + 1): 
    
    Z = np.random.standard_normal((N, 2))
    particles += u * h + np.sqrt(2 * D * h) * Z

    
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
       
        cp = ax.contourf(X_grid, Y_grid, C, levels=50, cmap='viridis')
        
        ax.set_title(f"Concentration at t = {step*h:.1f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(cp, ax=ax, label="C(x,t)")
        plot_idx += 1

plt.tight_layout()
plt.show()