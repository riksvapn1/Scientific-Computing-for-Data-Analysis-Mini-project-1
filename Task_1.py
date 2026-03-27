import numpy as np
import matplotlib.pyplot as plt

N = 2000 # num particles
T = 60 # time in seconds
h = 0.1 # step lenght
D = 0.02 # diffusion coefficient 
u = np.array([0.3,0])  # wind-vector

particles = np.zeros((N,2))# starting values particles

num_steps = int(T/h)

#-------fig setup---------
fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
axes = axes.flatten()  
plot_idx = 0


for step in range(1,num_steps+1): 
    Z = np.random.standard_normal((N,2))
    # Euler–Maruyama step: 
    particles += u*h + np.sqrt(2*D*h)*Z

    if step in[150,300,450,600]:
        t = step *h
        ax = axes[plot_idx]
        ax.scatter(particles[:, 0], particles[:, 1], s=1, alpha=0.6, color='blue')
        ax.set_title(f"Time: {t}s")
        ax.set_xlim(0, 25)  
        ax.set_ylim(-5, 5)  
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plot_idx +=1

plt.tight_layout()
plt.show()