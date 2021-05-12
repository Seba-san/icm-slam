import numpy as np
from copy import deepcopy as copy
#from ICM_SLAM import ConfigICM, ICM_external, Mapa, ICM_method
from ICM_SLAM import *



class ICM_external(ICM_method):
    """
    clase modificable por el usuario. Puede sobreescribir las funciones fun_x y
    fun_xn con el objetivo de determinar su propio funcional.
    """
    def __init__(self,config):
        ICM_method.__init__(self,config)


    def g(self, xt,ut):
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
        gg=xt+self.config.deltat*np.matmul(S,ut).reshape((3,1))
        return gg
    
    def h(self, xt,zt):
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
    
        y=self.mapa_visto
        alfa=zt[:,1]+xt[2]-np.pi/2.0
        zc=zt[:,0]*np.cos(alfa)
        zs=zt[:,0]*np.sin(alfa)
        # Resta la posicion de cada punto al mapa "visto". (y no es todo el mapa,
        # solo la parte matcheada en "actualizar mapa")
        distancias=np.concatenate((xt[0]+zc,xt[1]+zs)).reshape((len(alfa),2),order='F')-y
        # Calcula la norma :math:`hh^TQhh`
        aux=np.matmul(distancias,self.config.Q)
        potencial=np.sum(aux*distancias)
        return potencial

    def fun_xxn(self,x):
        """
        Esta función debe devolver un float para el cual su valor deve ser
        pequeño para posiciones de x más probables y un valor grande para
        valores improbables

        """
        a=1
        return 0

if __name__=='__main__':
    """
    Estructura general del código
     - lectura de datos
     - inicializar variables
     - iteracion 0
     - iteraciones ICM 
    """

    # Lectura de datos
    data=sio.loadmat('data_IJAC2018.mat')
    odometria = np.array(data['odometry'])#3x1833
    z = np.array(data['observations']) # 181x1833
    u = np.array(data['velocities'])#2x1833
    del(data)
    
    
    # Inicializacion de variables y objetos
    config=ConfigICM(z.shape[1])# Carga toda la configuración inicial
    #Filtro de observaciones
    zz=np.minimum(z+config.radio,z*0.0+config.rango_laser_max)

    #ICM=ICM_method(config) # Crea el objeto de los minimizadores
    ICM=ICM_external(config) # Crea el objeto de los minimizadores
    mapa_obj=Mapa(config)
    ICM.load_data(mapa_obj,zz,u,odometria)
    ##################### ITERACION ICM 0 #####################

    x=np.zeros((3,config.Tf))  #guarda la pose del DDMR en los Tf periodos de muestreo
    #1) Iteracion inicial ICM
    mapa_inicial,x=ICM.inicializar(x)

    # Preparacion de gráficos
    cambios_minimos=np.zeros(config.N)
    cambios_maximos=np.zeros(config.N)
    cambios_medios=np.zeros(config.N)
    graficar(x,mapa_inicial,odometria)#gráficos
    
    mapa_viejo=copy(mapa_inicial)
    
    #2) Iteraciones ICM
    for iteracionICM in range(config.N):
        print('iteración ICM : ',iteracionICM+1)
        mapa_refinado,x=ICM.itererar(mapa_viejo,x) #$2

        #CALCULO DE CAMBIOS
        [cambio_minimo,cambio_maximo,cambio_medio]=calc_cambio(mapa_refinado,mapa_viejo)
        cambios_minimos[iteracionICM]=cambio_minimo
        cambios_maximos[iteracionICM]=cambio_maximo
        cambios_medios[iteracionICM]=cambio_medio
        mapa_viejo=copy(mapa_refinado)
    
        graficar(x,mapa_refinado,odometria,iteracionICM)#gráficos

    graficar_cambio(cambios_minimos,cambios_maximos,cambios_medios)

