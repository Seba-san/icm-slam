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

class Sensor:
    """
    Esta clase esta en construccion, aún no se usa ni funciona.
    """
    def __init__(self,estructura):
        self.msgs=[]
        self.value==np.array([])
        self.estructura=estructura


    def call_back(self,msg):
        self.header_process(msg)

        D={}
        D['seq']=self.seq


    def header_process(self,msg):
        """
        uint32 seq
        time stamp
        string frame_id
        """
        self.seq=msg['header']['seq']
        
        pass



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

        self.laser=0
        self.odome=0

        self.debug=False

    def connect_ros(self):
        
        self.client = roslibpy.Ros(host='localhost', port=9090)
        #client.run_forever()
        self.client.run()
        if  self.client.is_connected:
            print('Conectado a la red ROS')
        else:
            self.disconnect_ros()

        listener_laser = roslibpy.Topic(self.client, self.config.topic_laser,
                self.config.topic_laser_msg)
        listener_laser.subscribe(self.callback_laser)

        listener_odometry = roslibpy.Topic(self.client, self.config.topic_odometry,
                self.config.topic_odometry_msg)
        listener_odometry.subscribe(self.callback_odometry)

        service = roslibpy.Service(self.client, '/icm_slam/iterative_flag','std_srvs/SetBool')
        service.advertise(self.icm_iterations_service)
         
    def disconnect_ros(self):
        print('modulo desconectado de la red ROS')
        self.client.terminate()
    

    def callback_laser(self,msg):
        """
        funcion de testeo
        """
        D={}
        D['seq']=msg['header']['seq']
        z=np.array([msg['ranges']],dtype=np.float)
        z=np.minimum(z+self.config.radio,z*0.0+self.config.rango_laser_max)
        D['data']=z.T
        self.laser_msg.append(D)
        self.principal_callback()

    def callback_laser_(self,msg):
        """
        LaserScan:angle_min, range_min, scan_time, range_max, angle_increment, angle_max,ranges,
        header, intensities.
        header: stamp, frame_id,seq
        """
        #import pdb; pdb.set_trace() # $3 sacar esto
        if self.seq0==0:
            self.seq0=msg['header']['seq']
        
        self.seq=msg['header']['seq']-self.seq0

        #if self.seq<self.mediciones.shape[1] and self.seq0>0: # same amount of odometry with laser items
        z=np.array([msg['ranges']],dtype=np.float)
        #z=z.T[0:-1:4] # subsampling from 720 to 180
        #z[np.isnan(z.astype('float'))]=self.config.rango_laser_max # clear None value
        z=np.minimum(z+self.config.radio,z*0.0+self.config.rango_laser_max)
        
        self.z=z.T
        self.laser=self.laser+1
        print('laser: ',self.laser,'sequencia: ',msg['header']['seq'])

    def callback_odometry(self,msg):
        """
        funcion de testeo
        """
        D={}
        D['seq']=msg['header']['seq']
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
        self.odometria_msg.append(copy(D))
        self.principal_callback()

    def callback_odometry_(self,msg):
        """
        'twist': {'twist': {'linear': {'y':, 'x':, 'z':}, 'angular': {'y':, 'x':, 'z':}},  'covariance':, 'header': 
        'pose': {'pose': {'position': {'y':, 'x':, 'z':}, 'orientation': {'y':, 'x':, 'z':, 'w':}},'covariance':, 'child_frame_id':}
        """

        #import pdb; pdb.set_trace() # $3 sacar esto
        if  self.seq0>0: # same amount of  odometry and laser items
           
            #self.seq>self.odometria.shape[1]
            msg_=copy(msg)
            msg=msg['pose']['pose']
            x=msg['position']['x']
            y=msg['position']['y']
            fi_x=msg['orientation']['x']
            fi_y=msg['orientation']['y']
            fi_z=msg['orientation']['z']
            fi_w=msg['orientation']['w']
            # Sacado de la libreria de cuaterniones
            # Fuente: https://github.com/Seba-san/pyquaternion
            t3 = +2.0 * (fi_w * fi_z + fi_x * fi_y)
            t4 = +1.0 - 2.0 * (fi_y **2 + fi_z **2)
            yaw_z = math.atan2(t3, t4)
            odo=np.array([[x,y,yaw_z]])
            #odo=np.array([[x,y,fi_w]])

            vx=msg_['twist']['twist']['linear']['x']
            vw=msg_['twist']['twist']['angular']['z']
            vel=np.array([[vx,vw]]).T
            #print('odo',odo)
            #print('vel:',vel)

            if not self.odometria.any():
                self.odometria=odo.T
                #self.x0=odo.T
                #self.u=vel
            else:
                self.odometria=np.hstack((self.odometria,odo.T))
                #self.u=np.hstack((self.u,vel))
            
            # Mediciones con frecuencia mayor a la odometria.
            
            #import pdb; pdb.set_trace() # $3 sacar esto
            if not self.mediciones.any():
                #if self.z.shape[0]==180:
                if self.z.shape[0]==181:
                    self.mediciones=self.z
            else:
                 self.mediciones=np.hstack((self.mediciones,self.z))

            if self.odometria.shape[1]>1:
                #self.new_data=self.new_data+1
                pass
            
            #self.new_data=self.new_data+1
        
        self.odome=self.odome+1
        print('odommetria: ',self.odome,'sequencia: ',msg_['header']['seq'])

    def principal_callback(self):
        num_odo=len(self.odometria_msg)
        num_laser=len(self.laser_msg)
        num_msg=min(num_odo,num_laser)
        #if not self.odometria.any():
        if not self.odometria.shape[0]>0:
            num_sensor=0
        else:
            num_sensor=self.odometria.shape[1]

        if num_msg>num_sensor:
            for t in range(num_sensor,num_msg):
                i=self.sort_sensors(t,self.laser_msg)
                laser=self.laser_msg[i]['data']
                i=self.sort_sensors(t,self.odometria_msg)
                odometria=copy(self.odometria_msg[i]['data']['odo'])
                u=copy(self.odometria_msg[i]['data']['u'])
                if self.odometria.shape[0]==0:
                    self.odometria=copy(odometria)
                    self.u=copy(u)
                    self.mediciones=laser
                else:
                    self.mediciones=np.hstack((self.mediciones,laser))
                    self.u=np.hstack((self.u,u))
                    self.odometria=np.hstack((self.odometria,odometria))

                self.new_data=self.new_data+1

    def sort_sensors(self,t,msg):
        if t==msg[t]['seq']-1:
            return t
        else:
            print('Estan llegando datos desordenados')
            L=len(msg)
            for i in range(L):
                if t==msg[i]['seq']-1:
                    return i

            print('mensaje: ',msg)
            print('t: ',t)
            print('Error 0: no se encuentra la secuencia buscada')
            sys.exit()

    def inicializar_online(self):

        xt=copy(self.x0)
        x=copy(self.x0)
        y=np.zeros((2,self.config.L)) #guarda la posicion de los a lo sumo L arboles del entorno
        self.mapa_obj=Mapa(self.config)

        self.connect_ros()
        k=10; # Intervalo de tiempo en segundos para mostrar mensaje.
        while self.new_data==0:
            pass

        #import pdb; pdb.set_trace() # $3 sacar esto
        t=time.time()
        z=filtrar_z(self.mediciones[:,-1],self.config)  #filtro la primer observacion [dist ang x y] x #obs
        zt=tras_rot_z(xt,z) #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,y,zt[:,2:4])
        dd=graficar2()#$5
        self.new_data=self.new_data-1
        self.t=1
        while time.time()<t+self.config.time: # Solo para test, luego incorporar el servicio
            if self.new_data>0:
                if x.shape[1]==41: # $9 DEBUGGER
                    self.debug=True
                else:
                    self.debug=False

                self.new_data=self.new_data-1
                y,xt=self.inicializar_online_process(y,xt)
                xt=np.reshape(xt,(3,1))

                #import pdb; pdb.set_trace() # $3 sacar esto
                x=np.concatenate((x,xt),axis=1)
                self.t=self.t+1
                if self.new_data>0:
                    print('Comparacion: ',x.shape[1]-self.odometria.shape[1])
                    print('Data: ',self.new_data)
                
                if (time.time()-t)>k:
                    k=k+10
                    print('Tiempo restante: ',self.config.time-(time.time()-t))
                
                dd.data(x,y,self.odometria)#$5

                if self.debug:
                    print('x actual: ',xt)


            if self.iterations_flag:
                print('Iterando para refinar los estados...')
                self.mapa_viejo=y
                self.positions=x
                break

        dd.show()#$5
        self.disconnect_ros()

        self.mapa_viejo=y
        self.positions=copy(x)

    def inicializar_online_process(self,y,xt):
        """
        callback del mensaje de ROS
        """
        #import pdb; pdb.set_trace() # $3 sacar esto
        t=self.t
        xtc=self.g(xt,self.u[:,t-1])  #actualizo cinemáticamente la pose
        #xtc=self.odometria[:,-1].reshape((3,1))
        z=self.filtrar_z(self.mediciones[:,t],self.config)  #filtro observaciones no informativas del tiempo t: [dist ang x y] x #obs
        if z.shape[0]==0:
            print('###0 !! Sin mediciones, abort')
            xt=xtc+0.0
            return y,xt.T
            #continue   #si no hay observaciones pasar al siguiente periodo de muestreo
        
        zt=tras_rot_z(xtc,z)  #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,y,zt[:,2:4])
        self.xt=copy(xt)
        xt=self.minimizar_x2(z[:,0:2],y[:,c].T)
        return y,xt

    def minimizar_x2(*arg):
        t=arg[0].t
        if arg[0].debug:
            print('Entradas a minimizar ',arg[1:])
            print('Odometira usada: ',arg[0].odometria[:,t-1:t+1])
            print('velocidad usada: ',arg[0].u[:,t-1])
            print('xt: ',arg[0].xt)

        a=arg[0].minimizar_x(*arg[1:])
        return a

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
        Callback service
        """
        
        xt=copy(self.x0)
        y=np.zeros((2,self.config.L)) #guarda la posicion de los a lo sumo L arboles del entorno
        self.mapa_obj.clear_obs()
        z=filtrar_z(self.mediciones[:,0],self.config)  #filtro la primer observacion [dist ang x y] x #obs
        odometria=self.odometria
        Tf=x.shape[1] 
        if z.shape[0]==0:
            print("## Sin mediciones 2")
            return mapa_viejo,x
            #continue #si no hay observaciones pasar al siguiente periodo de muestreo
        
        zt=tras_rot_z(xt,z) #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,mapa_viejo,zt[:,2:4])
        #BUCLE TEMPORAL
        dd=graficar2()#$5
        for t in range(1,Tf):
            z=filtrar_z(self.mediciones[:,t],self.config)  #filtro observaciones no informativas del tiempo t: [dist ang x y] x #obs
            if z.shape[0]==0:
                xt=(xt.reshape(3)+x[:,t+1])/2.0
                x[:,t]=copy(xt)
                print('skip por falta de obs')
                continue #si no hay observaciones pasar al siguiente periodo de muestreo
            
            #import pdb; pdb.set_trace() # $3 sacar esto
            zt=tras_rot_z(x[:,t],z)  #rota y traslada las observaciones de acuerdo a la pose actual
            y,c=self.mapa_obj.actualizar(y,mapa_viejo,zt[:,2:4])
            if t+1<Tf:
                xt=self.minimizar_xn(z[:,0:2],y[:,c].T,x,t)
            else:
                #import pdb; pdb.set_trace() # $3 sacar esto
                self.xt=(x[:,t]+0.0).reshape((3,1))
                xt=self.minimizar_x(z[:,0:2],y[:,c].T)

            x[:,t]=copy(xt)
            dd.data(x,y,self.odometria,N=7)#$5
        #filtro ubicaciones estimadas
        yy=self.mapa_obj.filtrar(y)
        yy=yy[:,:self.mapa_obj.landmarks_actuales]
        mapa_refinado=copy(yy)
        return mapa_refinado,x
    
    def filtrar_z(*arg):
        if arg[0].debug:
            print('filtrar z:', arg[1])

        a=filtrar_z(*arg[1:])
        return a
        
    def test_continuidad(self,x,xk):
        dt=0.1
        vel_max=0.2
        vel=(x[0:2]-xk[0:2])/dt
        vel_mag=np.linalg.norm(vel)
        c=vel_mag/(vel_max)
        if c>2: # evitar overflow
            return 500000

        f=np.e**(c**2)-1
        return f

    def g(self,xt,ut):
        if self.debug:
            print("en g. xt: ",xt," u: ",ut)

        xt=xt.reshape((3,1))
        ut=ut.reshape((2,1))
        S=np.array([[(np.cos(xt[2]))[0],0.0],[np.sin(xt[2])[0],0.0],[0.0,1.0]])
        xt_actualizado=xt+self.config.deltat*np.matmul(S,ut).reshape((3,1))
        return xt_actualizado.reshape((3,1))

    def minimizar_xn(self,medicion_actual,mapa_visto,x,t):
        self.x_ant=x[:,t-1].reshape((3,1))
        self.x_pos=x[:,t+1].reshape((3,1))
       
        self.xt=x[:,t].reshape((3,1))
        self.t=t
        self.medicion_actual=medicion_actual
        self.mapa_visto=mapa_visto
        x=fmin(self.fun_xn,(self.x_ant+self.x_pos)/2.0,xtol=0.001,disp=0)

        #import pdb; pdb.set_trace() # $3 sacar esto
        return x


    def minimizar_x(self,medicion_actual,mapa_visto):
        self.medicion_actual=medicion_actual
        self.mapa_visto=mapa_visto
        
        x0=self.g(self.xt,self.u[:,self.t-1])
        x=fmin(self.fun_x,x0,xtol=0.001,disp=0)
        #import pdb; pdb.set_trace() # $3 sacar esto
        return x

    def fun_x(self,x):
        # revisar
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
        u_ant=u_act[:,0]
        #odo=self.odometria[:,-2:]

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

        graficar(x,mapa_refinado,ICM.odometria)
        graficar_cambio(cambios_minimos,cambios_maximos,cambios_medios)
        import pdb; pdb.set_trace() # $3 sacar esto
