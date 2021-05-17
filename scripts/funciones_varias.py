
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist

def entrepi(angulo):
    """
    retorna el ángulo equivalente entre -pi y pi
    """
    angulo=np.mod(angulo,2*np.pi)
    if angulo>np.pi:
      angulo=angulo-2*np.pi
    
    return angulo

def tras_rot_z(x,z):
    """
     - Rota y traslada las observaciones de acuerdo a la pose actual
     - Transforma del body frame al global frame

    :math:`x_t=[ x_{t,x}, x_{t,y}, x_{t,\theta}]^T` 
    :math:`z_i=\{z_{t,i}:i=1,\cdots,n_t\}` :math:`z_{t,i}=[ z_{t,i,d}, z_{t,i,\theta} ]^T`

    """

    x=x.reshape((3,1))
    ct=np.cos(x[2]-np.pi/2.0)[0]
    st=np.sin(x[2]-np.pi/2.0)[0]
    R=np.array([[ct,st],[-st,ct]])
    z[:,2:4]=np.matmul(z[:,2:4],R)+np.matlib.repmat(x[0:2].T,z.shape[0],1)
    return z

def Rota(theta):
    """
    Arma la matriz de rotación en 2D a partir de un ángulo en *radianes*.
    """
    A=np.array([[np.cos(theta),np.sin(theta)],
        [-np.sin(theta),np.cos(theta)]])
    return A

def calc_cambio(y,mapa_viejo):
    min_dist=np.amin(cdist(mapa_viejo.T,y.T),axis=0)
    cambio_minimo=np.amin(min_dist)
    cambio_maximo=np.amax(min_dist)
    cambio_medio=np.mean(min_dist)
    return cambio_minimo,cambio_maximo,cambio_medio

def graficar(x,yy,odometria,N=0):
    plt.figure(N) 
    plt.plot(x[0],x[1], 'b')
    plt.plot(odometria[0],odometria[1], 'g')
    plt.plot(yy[0],yy[1], 'b*')
    plt.axis('equal')
    plt.show()

def graficar_cambio(cambios_minimos,cambios_maximos,cambios_medios):
    
    plt.figure(100) 
    plt.plot(cambios_minimos, 'b--')
    plt.plot(cambios_maximos, 'b--')
    plt.plot(cambios_medios, 'b')
    plt.show()

