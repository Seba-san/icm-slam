import numpy as np
import roslibpy # Conectarse a una red ROS
from ICM_SLAM import ICM_method,ConfigICM,Mapa, filtrar_z,tras_rot_z
from ICM_SLAM import graficar,graficar_cambio, calc_cambio,graficar2
from copy import deepcopy as copy
import time
import math
from funciones_varias import *

from scipy.optimize import fmin
import sys

from ros_test import Sensor

class Lidar(Sensor):
    def __init__(self,**argd):
        Sensor.__init__(self,**argd)

    def callback(self,msg):
        """
        LaserScan:angle_min, range_min, scan_time, range_max, angle_increment, angle_max,ranges,
        header, intensities.
        header: stamp, frame_id,seq
        """
        D={}
        D['seq']=msg['header']['seq']
        D['stamp']=msg['header']['stamp']['secs']+msg['header']['stamp']['nsecs']*10**(-9)
        D=self.header_process(msg)
        z=np.array([msg['ranges']],dtype=np.float)
        z[np.isnan(z.astype('float'))]=self.config.rango_laser_max # clear None value
        z=np.minimum(z+self.config.radio,z*0.0+self.config.rango_laser_max)
        if z.shape[1]!=180:
            angle_min=msg['angle_min']
            angle_increment=msg['angle_increment']
            s0=int((-np.pi/2-angle_min)/angle_increment)
            step=round((np.pi/180.0)/angle_increment)
            sfin=step*180
            z=z[:,s0:sfin:step]
            
        #import pdb; pdb.set_trace() # $3 sacar esto
        D['data']=z.T
        self.msgs.append(copy(D))
        self.principalCallback()

class Odometria(Sensor):
    def __init__(self,**argd):
        Sensor.__init__(self,**argd)

    def callback(self,msg):
        """
        'twist': {'twist': {'linear': {'y':, 'x':, 'z':}, 'angular': {'y':, 'x':, 'z':}},  'covariance':, 'header': 
        'pose': {'pose': {'position': {'y':, 'x':, 'z':}, 'orientation': {'y':, 'x':, 'z':, 'w':}},'covariance':, 'child_frame_id':}
        """
        D={}
        D['seq']=msg['header']['seq']
        D['stamp']=msg['header']['stamp']['secs']+msg['header']['stamp']['nsecs']*10**(-9)
        D=self.header_process(msg)

        #msg_=copy(msg)
        msg_=msg['pose']['pose']
        x=msg_['position']['x']
        y=msg_['position']['y']
        fi_x=msg_['orientation']['x']
        fi_y=msg_['orientation']['y']
        fi_z=msg_['orientation']['z']
        fi_w=msg_['orientation']['w']
        # Sacado de la libreria de cuaterniones
        # Fuente: https://github.com/Seba-san/pyquaternion
        t3 = +2.0 * (fi_w * fi_z + fi_x * fi_y)
        t4 = +1.0 - 2.0 * (fi_y **2 + fi_z **2)
        yaw_z = math.atan2(t3, t4)
        odo=np.array([[x,y,yaw_z]]).T

        vx=msg['twist']['twist']['linear']['x']
        vw=msg['twist']['twist']['angular']['z']
        vel=np.array([[vx,vw]]).T
        D['data']={'odo':odo,'u':vel}
        self.msgs.append(copy(D))
        #self.odometria_msg.append(copy(D))
        self.principalCallback()


class ICM_ROS(ICM_method):
    def __init__(self,config):
        ICM_method.__init__(self,config)
        self.new_data=0
        self.odometria=np.array([])
        self.mediciones=np.array([])
        self.u=np.array([])
        self.odometria_msg=[]
        self.laser_msg=[]
        self.u_msg=[]
    
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


    def connect_ros(self):
        """
        Crea los listeners y el service
        """
        
        self.client = roslibpy.Ros(host='localhost', port=9090)
        self.client.run()
        if  self.client.is_connected:
            print('Conectado a la red ROS')
        else:
            self.disconnect_ros()


        self.odom.subscribe(self.client)
        self.lidar.subscribe(self.client)
        """
        listener_laser = roslibpy.Topic(self.client, self.config.topic_laser,
                self.config.topic_laser_msg)
        listener_laser.subscribe(self.callback_laser)
        listener_odometry = roslibpy.Topic(self.client, self.config.topic_odometry,
                self.config.topic_odometry_msg)
        listener_odometry.subscribe(self.callback_odometry)
        """
        service = roslibpy.Service(self.client, '/icm_slam/iterative_flag','std_srvs/SetBool')
        service.advertise(self.icm_iterations_service)
         
    def disconnect_ros(self):
        print('modulo desconectado de la red ROS')
        self.client.terminate()

    def principal_callback(self):
        """
        A medida que esten todos los datos disponibles, va agregando al array
        de los sensores los datos de forma ordenada.
        """
        #print("Entrooooooooo")
        num_odo=len(self.odom.msgs)
        num_laser=len(self.odom.msgs)
        num_msg=min(num_odo,num_laser)
        if num_msg==0:
            return
        #if not self.odometria.any():
        # No importa que sensor se testee, hay la misma cantidad de datos
        # de cada sensor.
        if not self.odometria.shape[0]>0:
            num_sensor=0
            self.lidar.set_t0()
            self.odom.set_t0()
            #self.t0=self.laser_msg[0]['stamp'] # laser tiene data periodica.
        else:
            num_sensor=self.odometria.shape[1]
        
        if num_msg>num_sensor:
            for k in range(num_sensor,num_msg):
                #i1=self.lidar.sort(k)
                #i2=self.odom.sort(k)
                #i1=self.sort_sensors(k,self.laser_msg)
                #i2=self.sort_sensors(k,self.odometria_msg)

                laser,state1=self.lidar.sort(k)
                aux,state2=self.odom.sort(k)
                if not(state1 and state2):
                    continue

                odometria=aux['odo']
                u=aux['u']

                if self.odometria.shape[0]==0:
                    self.odometria=copy(odometria)
                    self.u=copy(u)
                    self.mediciones=laser
                else:
                    self.mediciones=np.hstack((self.mediciones,laser))
                    self.u=np.hstack((self.u,u))
                    self.odometria=np.hstack((self.odometria,odometria))

                self.new_data=self.new_data+1

    def sort_sensors(self,k,msg):
        """
        Busca dentro del historial de mensajes la secuencia correspondiente.
        Para agilizar la busqueda, si el valor inicial propuesto, coincide,
        rapidamente devuelve el indice.
        Ver si se puede hacer con el timestamp en lugar de hacerlo con el
        numero de secuencia.
        """
        ts=self.config.deltat
        now=k*ts+self.t0

        if abs(msg[k]['stamp']-now)<ts:
            return k
        else:
            print('Warning 0: hay un problema con los datos')
            L=len(msg)
            for i in range(L):
                 if abs(msg[i]['stamp']-now)<ts:
                     print('diferencia: ',i-k)
                     return i

            print('mensaje: ',msg[k])
            print('k: ',k)
            print('t0: ',self.t0)
            print('now: ',now)
            print('Error 0: no se encuentra la secuencia buscada')
            #sys.exit()
            return

    def inicializar_online(self):
        """
        Rutina principal que administra las iteraciones online. El nucleo se
        ejecuta en inicializar_online_process(), sin embargo hay muchas
        variables que hay que inicializar.
        """

        xt=copy(self.x0)
        x=copy(self.x0)
        y=np.zeros((2,self.config.L)) #guarda la posicion de los a lo sumo L arboles del entorno
        self.mapa_obj=Mapa(self.config)

        self.connect_ros()
        k=10; # Intervalo de tiempo en segundos para mostrar mensaje.
        while self.new_data==0:
            pass

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
                yy=self.mapa_obj.filtrar(y)
                yy=yy[:,:self.mapa_obj.landmarks_actuales]
                self.mapa_viejo=copy(yy)
                #self.mapa_viejo=y
                self.positions=x
                break
            
            if (time.time()-t)>k:
                k=k+10
                print('Tiempo restante: ',self.config.time-(time.time()-t))

        #dd.show()#$5
        self.disconnect_ros()
        yy=self.mapa_obj.filtrar(y)
        yy=yy[:,:self.mapa_obj.landmarks_actuales]
        self.mapa_viejo=copy(yy)

        #self.mapa_viejo=y
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

    def icm_iterations_service(self,request,response):
        """
        request['data']='start'
        """
        response['success']=True
        response['message']='Working...'
        self.iterations_flag=True
        return True

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
    import pdb; pdb.set_trace() # $3 sacar esto
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
        import pdb; pdb.set_trace() # $3 sacar esto
