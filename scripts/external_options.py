import numpy as np
from copy import deepcopy as copy
#from ICM_SLAM import ConfigICM, ICM_external, Mapa, ICM_method
from ICM_SLAM import *



class ICM_external(ICM_method):
    """
    clase modificable por el usuario. Puede sobreescribir las funciones fun_x
    fun_xn, g y h con el objetivo de determinar su propia configuracion.
    """
    def __init__(self,config):
        ICM_method.__init__(self,config)


    def g_(self,xt,ut):
        print('Se necesita definir la cinematica del vehículo')
        sys.exit()
        return np.zeros((3,1))

    def h_(self,xt,zt):
        print('Se necesita definir la funcion de medicion')
        sys.exit()
        return 0

    def fun_x_(self):
        print('Se necesita definir el argumento a minimizar para las iteraciones online')
        sys.exit()
        return 0

    def fun_xn_(self):
        print('Se necesita definir el argumento a minimizar para las iteraciones offline')
        sys.exit()
        return 0

if __name__=='__main__':
    """
    Estructura general del código
     - lectura de datos
     - inicializar variables
     - iteracion 0
     - iteraciones ICM 
    """

    # Inicializacion de variables y objetos
    config=ConfigICM('config_ros.yaml')# Carga toda la configuración inicial
    # Lectura de datos
    #data=sio.loadmat(config.file)
    #odometria = np.array(data['odometry'])#3x1833
    #z = np.array(data['observations']) # 181x1833
    #u = np.array(data['velocities'])#2x1833
    #del(data)
    
    #Filtro de observaciones
    zz=np.minimum(z+config.radio,z*0.0+config.rango_laser_max)

    #ICM=ICM_method(config) # Crea el objeto de los minimizadores
    #config.set_Tf(z.shape[1]) 
    ICM=ICM_external(config) # Crea el objeto de los minimizadores
    mapa_obj=Mapa(config)
    ICM.load_data(mapa_obj,zz,u,odometria)
    ##################### ITERACION ICM 0 #####################

    #1) Iteracion inicial ICM
    #"""
    x=np.zeros((3,config.Tf))  #guarda la pose del DDMR en los Tf periodos de muestreo
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
    #"""
