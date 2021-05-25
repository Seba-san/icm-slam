import numpy as np
import roslibpy # Conectarse a una red ROS
from ICM_SLAM import ICM_method,ConfigICM,Mapa, filtrar_z,tras_rot_z
from ICM_SLAM import graficar,graficar_cambio, calc_cambio,graficar2
from copy import deepcopy as copy
import time
import math


class ICM_ROS(ICM_method):
    def __init__(self,config):
        ICM_method.__init__(self,config)
        self.new_data=False
        self.odometria=np.array([])
        self.mediciones=np.array([])
    
        self.iterations_flag=False
        self.seq0=0
        self.seq=0

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
         
        """
        try:
               while True:
                   pass
        except KeyboardInterrupt:
               client.terminate()
        """
    def disconnect_ros(self):
        print('modulo desconectado de la red ROS')
        self.client.terminate()

    def callback_laser(self,msg):
        """
        LaserScan:angle_min, range_min, scan_time, range_max, angle_increment, angle_max,ranges,
        header, intensities.
        header: stamp, frame_id,seq
        """
        if self.seq0==0:
            self.seq0=msg['header']['seq']
        
        self.seq=msg['header']['seq']-self.seq0

        #if self.seq<self.mediciones.shape[1] and self.seq0>0: # same amount of odometry with laser items
        z=np.array([msg['ranges']],dtype=np.float)
        z=z.T[0:-1:4] # subsampling from 720 to 180
        z[np.isnan(z.astype('float'))]=self.config.rango_laser_max # clear None value
        z=np.minimum(z+self.config.radio,z*0.0+self.config.rango_laser_max)
        
        self.z=z

    def callback_odometry(self,msg):
        """
        'twist': {'twist': {'linear': {'y':, 'x':, 'z':}, 'angular': {'y':, 'x':, 'z':}},  'covariance':, 'header': 
        'pose': {'pose': {'position': {'y':, 'x':, 'z':}, 'orientation': {'y':, 'x':, 'z':, 'w':}},'covariance':, 'child_frame_id':}
        """

        if  self.seq0>0: # same amount of  odometry and laser items
           
            #self.seq>self.odometria.shape[1]
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
            
            if not self.odometria.any():
                self.odometria=odo.T
                self.x0=odo.T
            else:
                self.odometria=np.hstack((self.odometria,odo.T))
            
            # Mediciones con frecuencia mayor a la odometria.
            
            #import pdb; pdb.set_trace() # $3 sacar esto
            if not self.mediciones.any():
                if self.z.shape[0]==180:
                    self.mediciones=self.z
            else:
                 self.mediciones=np.hstack((self.mediciones,self.z))
                 #self.odometria=np.concatenate((self.odometria,odo.T),axis=1)

            if self.odometria.shape[1]>1:
                self.new_data=True
        
    def inicializar_online(self):

        xt=copy(self.x0)
        x=np.array([])
        y=np.zeros((2,self.config.L)) #guarda la posicion de los a lo sumo L arboles del entorno
        self.mapa_obj=Mapa(config)

        self.connect_ros()
        t=time.time()
        k=10; # Intervalo de tiempo en segundos para mostrar mensaje.
        
        dd=graficar2()#$5
        while time.time()<t+self.config.time: # Solo para test, luego incorporar el servicio
            if self.new_data:
                self.new_data=False
                y,xt=self.inicializar_online_process(y)
                xt=np.reshape(xt,(3,1))
                if not x.any():
                    x=copy(self.x0)

                x=np.concatenate((x,xt),axis=1)
                if (time.time()-t)>k:
                    k=k+10
                    print('Tiempo restante: ',self.config.time-(time.time()-t))
                
                dd.data(x,y,self.odometria)#$5
                #mapa_inicial,x=ICM.inicializar(x)
            if self.iterations_flag:
                print('Iterando para refinar los estados...')
                self.mapa_viejo=y
                self.positions=x
                break

        dd.show()#$5
        #graficar(x,y,self.odometria)
        self.disconnect_ros()

        self.mapa_viejo=y
        self.positions=copy(x)

    def inicializar_online_process(self,y):
        """
        callback del mensaje de ROS
        """
        #xtc=self.g(xt,u[:,t-1])  #actualizo cinemáticamente la pose
        xtc=self.odometria[:,-1].reshape((3,1))
        z=filtrar_z(self.mediciones[:,-1],self.config)  #filtro observaciones no informativas del tiempo t: [dist ang x y] x #obs
        if z.shape[0]==0:
            print('###!! Sin mediciones, abort')
            xt=xtc+0.0
            return y,xt
            #continue   #si no hay observaciones pasar al siguiente periodo de muestreo
        
        zt=tras_rot_z(xtc,z)  #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,y,zt[:,2:4])
        self.xt=xtc
        xt=self.minimizar_x(z[:,0:2],y[:,c].T)
        return y,xt

    def icm_iterations_service(self,request,response):
        """
        request['data']='start'
        """
        #print('Service called: ', request)
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
            
            import pdb; pdb.set_trace() # $3 sacar esto
            zt=tras_rot_z(x[:,t],z)  #rota y traslada las observaciones de acuerdo a la pose actual
            y,c=self.mapa_obj.actualizar(y,mapa_viejo,zt[:,2:4])
            if t+1<Tf:
                xt=self.minimizar_xn(z[:,0:2],y[:,c].T,x,t)
            else:
                #import pdb; pdb.set_trace() # $3 sacar esto
                self.xt=copy(x[:,t])
                xt=self.minimizar_x(z[:,0:2],y[:,c].T)

            x[:,t]=copy(xt)
            dd.data(x,y,self.odometria,N=7)#$5
        #filtro ubicaciones estimadas
        yy=self.mapa_obj.filtrar(y)
        yy=yy[:,:self.mapa_obj.landmarks_actuales]
        mapa_refinado=copy(yy)
        return mapa_refinado,x

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

if __name__=='__main__':
    config=ConfigICM('config_ros.yaml')
    ICM=ICM_ROS(config)
    ICM.inicializar_online()
    ICM.iterations_flag=True # Borrar esto
    if ICM.iterations_flag:
        mapa_viejo=copy(ICM.mapa_viejo)
        x=copy(ICM.positions)

        cambios_minimos=np.zeros(config.N)
        cambios_maximos=np.zeros(config.N)
        cambios_medios=np.zeros(config.N)
        
        dd=graficar2()#$5
        for iteracionICM in range(config.N):
            print('iteración ICM : ',iteracionICM+1)
            mapa_refinado,x=ICM.iterations_process_offline(mapa_viejo,x) #$2
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
