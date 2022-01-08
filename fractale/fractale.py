
import pandas as pd
import numpy as np
from PIL import Image

fond=Image.new('RGB', (100,200,200))
fond.show()


"""Soit la suite (Zn)n de complexes : Zo=0, Zn+1=Zn^2 + C, C un complexe"""
import numpy as np
import matplotlib.pyplot as plt

c=10+10.j
n=600

def julia(c,n):
    suite=[0]
    u=0
    for k in range(n):
        u=u**2 + c
        suite.append(u)
    return(suite)

def duree_de_vie(c):
    u=c
    for j in range(100):
        u = u**2 + c
        if np.abs(u) > 2:
            return(j + 1)
    return(0)
    
z=n
Z = np.zeros((z,z), dtype = np.complex) # si on met le dtype, python/numpy considère qu'on a une matrice réelle donc les imaginaires ne rentrent pas
for i, reel in enumerate(np.linspace(-2,2,z)):
    for k, image in enumerate(np.linspace(2,-2,z)):
        Z[i,k] = reel + image*1j
        
"""print(Z)
plt.scatter(Z.real, Z.imag)"""

I = np.zeros(shape = (n,n))
for i in range(n):
    for j in range(n):
        I[i,j] = duree_de_vie(Z[i,j])
        
plt.figure(figsize=(15,15))
plt.imshow(I, cmap='hot') # cmap permet de changer la couleur de base
plt.colorbar() # sert à avoir la barre sur le côté des couleurs


# points : -0.74-0.1j avec 10^-3 de zoom donne un truc sympa !!!!!