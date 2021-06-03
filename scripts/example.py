from ICM_ROS import ICM_ROS
from ICM_SLAM_tools import ConfigICM,graficar2
from copy import deepcopy as copy

# Librerias optativas
import numpy as np
import sys


"""
Ejemplo de como modificar el comportamiento
"""
class My_method(ICM_ROS):
    def __init__(self,config):
        ICM_ROS.__init__(self,config)

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

    # ========= Principal line
    config=ConfigICM('config_ros.yaml')
    ICM=ICM_ROS(config)
    ICM.inicializar_online()
    # ========= Principal line
    if ICM.iterations_flag:
        mapa_viejo=copy(ICM.mapa_viejo)
        x=copy(ICM.positions)
        
        dd=graficar2()#$5
        for iteracionICM in range(config.N):
            print('iteración ICM : ',iteracionICM+1)
            # ========= Principal line
            mapa_refinado,x=ICM.iterations_process_offline(mapa_viejo,x) #$2
            # ========= Principal line
            dd.data(x,mapa_refinado,ICM.odometria,N=11)#$5
