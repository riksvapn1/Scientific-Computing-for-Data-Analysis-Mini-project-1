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


# ---Grid Setup ---
x_range = np.linspace(0, 25, 200)
y_range = np.linspace(-5, 5, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range) # X and Y_grid same size


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten() 
plot_idx = 0
norm_factor = 1 / (2 * np.pi * eps**2)
# ---  sim loop ---
for step in range(1, num_steps + 1): 
    Z = np.random.standard_normal((N,2))
    # Euler–Maruyama step: 
    particles[0: N, :] += u*h + np.sqrt(2*D*h)*Z
    N += int(Q*h) # new amount of particles per step
    
    if step in [150, 300, 450, 600]:
        C = np.zeros_like(X_grid)
        
        for k in range(N):
            xp, yp = particles[k, 0], particles[k, 1] # every particle coord
            # Distance squared (x^T x)
            dist_sq = (X_grid- xp)**2 + (Y_grid- yp)**2
            
            C+= norm_factor * np.exp(-dist_sq / (2 * eps**2))
        # Average the field
        C/= N
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
