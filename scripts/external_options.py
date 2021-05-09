import numpy as np
import yaml


class ConfigICM:
    """
    Núcleo de todos los parámetros de configuración y constantes. 
    """
    def __init__(self,Tf,configFile='config_default.yaml',D={}):
        if not D:
            arch=open(configFile, 'r')
            Data = yaml.load(arch,Loader=yaml.FullLoader)
            D=Data['D']
            

        # parámetros por default
        self.N=D['N']  #cantidad de iteraciones de ICM
        self.deltat=D['deltat']  #periodo de muestreo
        self.Tf=Tf  #cantidad total de periodos de muestreo
        self.L=D['L']  #cota superior de la cantidad de objetos (max landmarks)
        
        self.Q=np.eye(2)#matriz de covarianza de las observaciones
        self.Q[0,0]=D['Q'][0] 
        self.Q[1,1]=D['Q'][1] 
        
        self.R=np.eye(3) #matriz de covarianza del motion model
        self.R[0,0]=D['R'][0]
        self.R[1,1]=D['R'][1]
        self.R[2,2]=D['R'][2]

        self.cte_odom=D['cte_odom']  #S=diag([cte_odom,cte_odom,cte_odom]) matriz de peso de los datos odométricos
        self.cota=D['cota']  #cantidad de veces que hay q ver un arbol para q se considere un arbol
        self.dist_thr=D['dist_thr']  #distancia máxima para que dos obs sean consideradas del mismo objeto
        self.dist_thr_obs=D['dist_thr_obs']  #distancia máxima para que dos obs sean consideradas del mismo objeto en el proceso de filtrado de las observaciones
        self.rango_laser_max=D['rango_laser_max']  #alcance máximo del laser
        self.radio=D['radio'] #radio promedio de los árboles

def entrepi(angulo):
    """
    retorna el ángulo equivalente entre -pi y pi
    """
    angulo=np.mod(angulo,2*np.pi)
    if angulo>np.pi:
      angulo=angulo-2*np.pi
    
    return angulo

def g(xt,ut, config):
    """
    Modelo de la odometria
    ========================
    Actualización de la pose a partir de las señales de control usando solo la
    cinemática del vehículo. Como referencia revisar definicion en el paper,
    página 8/15. 
    Entradas:
     - :math:`x_t=[ x_{t,x}, x_{t,y}, x_{t,\theta}]^T` 
     - :math:`u_t=[ v_t, \omega_t ]^T`
     - config: Objeto que contiene todas las configuraciones

    Salida
     - :math:`x_{t+1}=[ x_{t+1,x}, x_{t+1,y}, x_{t+1,\theta}]^T`

     A futuro poner la ecuación en este lugar.
    """

    xt=xt.reshape((3,1))
    ut=ut.reshape((2,1))
    S=np.array([[(np.cos(xt[2]))[0],0.0],
        [np.sin(xt[2])[0],0.0],
        [0.0,1.0]])
    gg=xt+config.deltat*np.matmul(S,ut).reshape((3,1))
    return gg

def h(xt,zt,ICM):
    """
    Función de potencial energético debido a las observaciones y al mapa. 

    Modelo de las observaciones
    =============================
    Página 8/15 del paper.
    Modelo de las observaciones para el Laser 2D
    Entradas:
     - :math:`x_t=[ x_{t,x}, x_{t,y}, x_{t,\theta}]^T` 
     - :math:`z_i=\{z_{t,i}:i=1,\cdots,n_t\}` :math:`z_{t,i}=[ z_{t,i,d}, z_{t,i,\theta} ]^T`
     - config: Objeto que contiene todas las configuraciones

    Salida:
     -  Potencial.

     Poner la forma de la función que esta en el paper.
    """

    y=ICM.yopt
    alfa=zt[:,1]+xt[2]-np.pi/2.0
    zc=zt[:,0]*np.cos(alfa)
    zs=zt[:,0]*np.sin(alfa)
    # Resta la posicion de cada punto al mapa "visto". (y no es todo el mapa,
    # solo la parte matcheada en "actualizar mapa")
    distancias=np.concatenate((xt[0]+zc,xt[1]+zs)).reshape((len(alfa),2),order='F')-y
    # Calcula la norma :math:`hh^TQhh`
    aux=np.matmul(distancias,ICM.config.Q)
    potencial=np.sum(aux*distancias)
    return potencial

