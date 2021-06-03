import numpy as np
#import roslibpy # Conectarse a una red ROS
#from ICM_SLAM import ROS,ConfigICM,Mapa, filtrar_z,tras_rot_z,Sensor
#from ICM_SLAM import graficar,graficar_cambio, calc_cambio,graficar2
from ICM_SLAM_tools import *
from copy import deepcopy as copy
import time
import math
from funciones_varias import *

from scipy.optimize import fmin
import sys
from sensors_definitions import Lidar, Odometria

class ICM_ROS(ROS):
    def __init__(self,config,x0=''):
        #ICM_method.__init__(self,config)
        super().__init__()
        if x0=='':
            self.x0=np.zeros((3,1))  #guarda la pose actual (inicial en esta linea) del DDMR
        else:
            self.x0=x0

        self.config=config
        self.new_data=0
        self.odometria=np.array([])
        self.mediciones=np.array([])
        self.u=np.array([])
    
        self.iterations_flag=False
        self.seq0=0
        self.seq=0

        self.debug=False

        D={}
        D['config']=config
        D['principalCallback']=self.principal_callback

        # sensors implementation
        D['name']='lidar'
        D['topic']=self.config.topic_laser
        D['topic_msg']=self.config.topic_laser_msg
        self.lidar=Lidar(**D)
        
        D['name']='odometria'
        D['topic']=self.config.topic_odometry
        D['topic_msg']=self.config.topic_odometry_msg
        self.odom=Odometria(**D)

    def inicializar_online(self):
        """
        Rutina principal que administra las iteraciones online. El nucleo se
        ejecuta en inicializar_online_process(), sin embargo hay muchas
        variables que hay que inicializar.
        """

        self.connect_ros()
        while self.new_data==0:
            pass
        self.x0=np.array([self.odometria[:,0]]).T

        xt=copy(self.x0)
        x=copy(self.x0)
        y=np.zeros((2,self.config.L)) #guarda la posicion de los a lo sumo L arboles del entorno
        self.mapa_obj=Mapa(self.config)

        k=10; # Intervalo de tiempo en segundos para mostrar mensaje.

        t=time.time()
        z=filtrar_z(self.mediciones[:,-1],self.config)  #filtro la primer observacion [dist ang x y] x #obs
        zt=tras_rot_z(xt,z) #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,y,zt[:,2:4])
        dd=graficar2()#$5
        self.new_data=self.new_data-1
        self.t=1
        while time.time()<t+self.config.time: # Solo para test, luego incorporar el servicio
            if self.new_data>0:

                self.new_data=self.new_data-1
                y,xt=self.inicializar_online_process(y,xt)
                xt=np.reshape(xt,(3,1))

                x=np.concatenate((x,xt),axis=1)
                self.t=self.t+1
            else:
                
                dd.data(x,y,self.odometria)#$5

            if self.iterations_flag and self.new_data==0:
                print('Iterando para refinar los estados...')
                break
            
            if (time.time()-t)>k:
                k=k+10
                print('Tiempo restante: ',self.config.time-(time.time()-t))

        #dd.show()#$5
        self.disconnect_ros()
        yy=self.mapa_obj.filtrar(y)
        yy=yy[:,:self.mapa_obj.landmarks_actuales]
        self.mapa_viejo=copy(yy)

        self.positions=copy(x)

    def inicializar_online_process(self,y,xt):
        """
        Nucleo del proceso inicializacion online.
        """
        t=self.t
        xtc=self.g(xt,self.u[:,t-1])  #actualizo cinemáticamente la pose
        #xtc=self.odometria[:,-1].reshape((3,1))
        z=filtrar_z(self.mediciones[:,t],self.config)  #filtro observaciones no informativas del tiempo t: [dist ang x y] x #obs
        if z.shape[0]==0:
            #print('###0 !! Sin mediciones, abort')
            xt=xtc+0.0
            return y,xt.T#si no hay observaciones pasar al siguiente periodo de muestreo

        zt=tras_rot_z(xtc,z)  #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,y,zt[:,2:4])
        self.xt=copy(xt)
        xt=self.minimizar_x(z[:,0:2],y[:,c].T)
        return y,xt

    def iterations_process_offline(self,mapa_viejo,x):
        """
        Proceso de iteraciones offline. Itera sobre toda la secuencia. Como
        argumentos de entrada son el "mapa inicial" y los estados "x". Como
        salida devuelve el mapa refinado y las poses refinadas.  
        """
        xt=copy(self.x0)
        y=np.zeros((2,self.config.L)) #guarda la posicion de los a lo sumo L arboles del entorno
        self.mapa_obj.clear_obs()
        z=filtrar_z(self.mediciones[:,0],self.config)  #filtro la primer observacion [dist ang x y] x #obs
        odometria=self.odometria
        Tf=x.shape[1] 
        if z.shape[0]==0:
            #print("## Sin mediciones 2")
            return mapa_viejo,x#si no hay observaciones pasar al siguiente periodo de muestreo

        zt=tras_rot_z(xt,z) #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,mapa_viejo,zt[:,2:4])
        #BUCLE TEMPORAL
        #dd=graficar2()#$5
        for t in range(1,Tf):
            z=filtrar_z(self.mediciones[:,t],self.config)  #filtro observaciones no informativas del tiempo t: [dist ang x y] x #obs
            if z.shape[0]==0:
                xt=(xt.reshape(3)+x[:,t+1])/2.0
                x[:,t]=copy(xt)
                #print('skip por falta de obs')
                continue #si no hay observaciones pasar al siguiente periodo de muestreo
            
            zt=tras_rot_z(x[:,t],z)  #rota y traslada las observaciones de acuerdo a la pose actual
            y,c=self.mapa_obj.actualizar(y,mapa_viejo,zt[:,2:4])
            if t+1<Tf:
                xt=self.minimizar_xn(z[:,0:2],y[:,c].T,x,t)
            else:
                self.xt=(x[:,t-1]+0.0).reshape((3,1))
                self.t=t
                xt=self.minimizar_x(z[:,0:2],y[:,c].T)

            x[:,t]=copy(xt)
            #dd.data(x,y,self.odometria,N=7)#$5
        #filtro ubicaciones estimadas
        yy=self.mapa_obj.filtrar(y)
        yy=yy[:,:self.mapa_obj.landmarks_actuales]
        mapa_refinado=copy(yy)
        return mapa_refinado,x
    
    """
    Todas las funciones siguientes son las que puede configurar el usuario
    inicialmente.
    """

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

    def g(self,xt,ut):
        xt=xt.reshape((3,1))
        ut=ut.reshape((2,1))
        S=np.array([[(np.cos(xt[2]))[0],0.0],[np.sin(xt[2])[0],0.0],[0.0,1.0]])
        xt_actualizado=xt+self.config.deltat*np.matmul(S,ut).reshape((3,1))
        return xt_actualizado.reshape((3,1))

    def minimizar_xn(self,medicion_actual,mapa_visto,x,t):
        self.x_ant=x[:,t-1].reshape((3,1))
        self.x_pos=x[:,t+1].reshape((3,1))
       
        self.xt=x[:,t-1].reshape((3,1))
        self.t=t
        self.medicion_actual=medicion_actual
        self.mapa_visto=mapa_visto
        x=fmin(self.fun_xn,(self.x_ant+self.x_pos)/2.0,xtol=0.001,disp=0)
        return x

    def fun_xn(self,x):
        t=self.t
        z=self.medicion_actual
        x_pos=self.x_pos
        u_act=self.u[:,t-1:t+1]
        odo=self.odometria[:,t-1:t+2]
        
        #f=self.fun_x(x)
        x=x.reshape((3,1))
        gg=self.g(x,u_act[:,1])-x_pos
        gg[2]=entrepi(gg[2])
        Rotador=Rota(x[2][0])
        ooo=np.zeros((3,1))
        ooo[0:2]=np.matmul(Rota(odo[2,1]),(odo[0:2,2]-odo[0:2,1]).reshape((2,1)))-np.matmul(Rotador,x_pos[0:2]-x[0:2])
        ooo[2]=odo[2,2]-odo[2,1]-x_pos[2]+x[2]
        ooo[2]=entrepi(ooo[2])
        f=np.matmul(np.matmul(gg.T,self.config.R),gg)+self.config.cte_odom*np.matmul(ooo.T,ooo)

        x_ant=self.xt
        # Cual de los 2? no son lo mismo
        u_ant=u_act[:,0]
        #u_ant=uself.u[:,t-1]

        gg=x.reshape((3,1))-self.g(x_ant,u_ant)
        gg[2]=entrepi(gg[2])
        hh=self.h(x,z)
        Rotador=Rota(x_ant[2][0])
        ooo=np.zeros((3,1))
        ooo[0:2]=np.matmul(Rota(odo[2,0]),(odo[0:2,1]-odo[0:2,0]).reshape((2,1)))-np.matmul(Rotador,x[0:2].reshape((2,1))-x_ant[0:2])
        ooo[2]=odo[2,1]-odo[2,0]-x[2]+x_ant[2]
        ooo[2]=entrepi(ooo[2])
        f=f+np.matmul(np.matmul(gg.T,self.config.R),gg)+hh+self.config.cte_odom*np.matmul(ooo.T,ooo)
        return f

    def minimizar_x(self,medicion_actual,mapa_visto):
        self.medicion_actual=medicion_actual
        self.mapa_visto=mapa_visto
        
        x0=self.g(self.xt,self.u[:,self.t-1])
        x=fmin(self.fun_x,x0,xtol=0.001,disp=0)
        return x

    def fun_x(self,x):
        t=self.t
        z=self.medicion_actual
        x_ant=self.xt
        u_ant=self.u[:,t-1]
        odo=self.odometria[:,t-1:t+1]

        gg=x.reshape((3,1))-self.g(x_ant,u_ant)
        gg[2]=entrepi(gg[2])
        hh=self.h(x,z)
        Rotador=Rota(x_ant[2][0])
        ooo=np.zeros((3,1))
        ooo[0:2]=np.matmul(Rota(odo[2,0]),(odo[0:2,1]-odo[0:2,0]).reshape((2,1)))-np.matmul(Rotador,x[0:2].reshape((2,1))-x_ant[0:2])
        ooo[2]=odo[2,1]-odo[2,0]-x[2]+x_ant[2]
        ooo[2]=entrepi(ooo[2])
        f=np.matmul(np.matmul(gg.T,self.config.R),gg)+hh+self.config.cte_odom*np.matmul(ooo.T,ooo)
        return f

if __name__=='__main__':
    
    # ========= Principal line
    config=ConfigICM('config_ros.yaml')
    ICM=ICM_ROS(config)
    ICM.inicializar_online()
    # ========= Principal line
    ICM.iterations_flag=True # Borrar esto
    #import pdb; pdb.set_trace() # $3 sacar esto
    if ICM.iterations_flag:
        mapa_viejo=copy(ICM.mapa_viejo)
        x=copy(ICM.positions)

        cambios_minimos=np.zeros(config.N)
        cambios_maximos=np.zeros(config.N)
        cambios_medios=np.zeros(config.N)
        
        dd=graficar2()#$5
        for iteracionICM in range(config.N):
            print('iteración ICM : ',iteracionICM+1)
            # ========= Principal line
            mapa_refinado,x=ICM.iterations_process_offline(mapa_viejo,x) #$2
            # ========= Principal line
            print('Correccion: ', np.linalg.norm(x-ICM.positions,axis=1).sum())
            dd.data(x,mapa_refinado,ICM.odometria,N=11)#$5

            #CALCULO DE CAMBIOS
            [cambio_minimo,cambio_maximo,cambio_medio]=calc_cambio(mapa_refinado,mapa_viejo)
            cambios_minimos[iteracionICM]=cambio_minimo
            cambios_maximos[iteracionICM]=cambio_maximo
            cambios_medios[iteracionICM]=cambio_medio
            mapa_viejo=copy(mapa_refinado)

        print('Cerrar las imagenes para continuar')
        graficar(x,mapa_refinado,ICM.odometria)
        graficar_cambio(cambios_minimos,cambios_maximos,cambios_medios)
        #import pdb; pdb.set_trace() # $3 sacar esto
