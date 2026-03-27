import numpy as np 
import matplotlib.pyplot as plt

# ====Task 1 replica========

# --Parameters ---
N = 0 
T = 60 
h = 0.1 
D = 0.02 
u = np.array([0.3, -0.06])  
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
    N += int(Q*h)
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


# ======Task 2 replica======