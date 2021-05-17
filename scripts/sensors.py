import numpy as np
import roslibpy # Conectarse a una red ROS
from ICM_SLAM import ICM_method,ConfigICM,Mapa, filtrar_z,tras_rot_z
from ICM_SLAM import graficar
from copy import deepcopy as copy
import time
import math


class ICM_ROS(ICM_method):
    def __init__(self,config):
        ICM_method.__init__(self,config)
        self.new_data=False
        self.odometria=copy(self.x0)
        self.mediciones=np.zeros((180,1))# depende de la configuracion del sensor
    
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
         
        """
        try:
               while True:
                   pass
        except KeyboardInterrupt:
               client.terminate()
        """
    def disconnect_ros(self):
        print('No se puedo conectar a la red ROS')
        self.client.terminate()

 
    def callback_laser(self,msg):
        """
        LaserScan:angle_min, range_min, scan_time, range_max, angle_increment, angle_max,ranges,
        header, intensities.
        header: stamp, frame_id,seq
        """
        #print('Heard talking: ')
       # s=msg['header']['stamp']['secs']
       # ns=msg['header']['stamp']['nsecs']
       # timestamp=float(s+ns*10**-9)
       # stamp=msg['header']['seq']
       # print('laser seq: ',stamp)
        if self.seq0==0:
            self.seq0=msg['header']['seq']
        
        self.seq=msg['header']['seq']-self.seq0

        #if self.seq<self.mediciones.shape[1] and self.seq0>0: # same amount of odometry with laser items
        z=np.array([msg['ranges']],dtype=np.float)
        z=z.T[0:-1:4] # subsampling from 720 to 180
        z[np.isnan(z.astype('float'))]=self.config.rango_laser_max # clear None value
        z=np.minimum(z+self.config.radio,z*0.0+self.config.rango_laser_max)
        #if self.mediciones==[]:
        #    self.mediciones=z
        #else:
        #print('###!! z shape : ',z.shape)
        #print('###!! z  : ',z)
        
        self.mediciones=np.concatenate((self.mediciones,z),axis=1)

        #self.new_data=True
        #print('##!! Seq: ',self.seq)

    def callback_odometry(self,msg):
        """
        'twist': {'twist': {'linear': {'y':, 'x':, 'z':}, 'angular': {'y':, 'x':, 'z':}},  'covariance':, 'header': 
        'pose': {'pose': {'position': {'y':, 'x':, 'z':}, 'orientation': {'y':, 'x':, 'z':, 'w':}},'covariance':, 'child_frame_id':}
        """

        if self.seq>self.odometria.shape[1] and self.seq0>0: # same amount of odometry with laser items
            msg=msg['pose']['pose']
            #print('#### ',msg['position'])
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
            self.odometria=np.concatenate((self.odometria,odo.T),axis=1)

            self.new_data=True
            #print(self.odometria)
            #print('odometry shape: ',self.odometria.shape[1])
            #print('##!! Seq: ',self.seq)
        
    def inicializar_online(self):

        xt=copy(self.x0)
        x=copy(self.x0)
        y=np.zeros((2,self.config.L)) #guarda la posicion de los a lo sumo L arboles del entorno
        self.mapa_obj=Mapa(config)


        self.connect_ros()
        t=time.time()
        k=10;
        while time.time()<t+300.0: # Solo para test, luego incorporar el servicio
            if self.new_data:
                self.new_data=False
                y,xt=self.inicializar_online_process(y,xt)
                xt=np.reshape(xt,(3,1))
                #print(xt)
                x=np.concatenate((x,xt),axis=1)
                if (time.time()-t)>k:
                    k=k+10
                    print('Tiempo restante: ',300.0- (time.time()-t))
                
                #mapa_inicial,x=ICM.inicializar(x)

        graficar(x,y,self.odometria)
        self.disconnect_ros()

        #self.itererar(mapa_viejo,x)

    def inicializar_online_process(self,y,xt):
        """
        callback del servicio de ROS
        """

        #xtc=self.g(xt,u[:,t-1])  #actualizo cinemáticamente la pose
        xtc=self.odometria[:,-1].reshape((3,1))
        z=filtrar_z(self.mediciones[:,-1],self.config)  #filtro observaciones no informativas del tiempo t: [dist ang x y] x #obs
        if z.shape[0]==0:
            #print('###!! Sin mediciones, abort')
            xt=xtc
            #x[:,t]=xt.T
            return y,xt
            #continue   #si no hay observaciones pasar al siguiente periodo de muestreo
        
        zt=tras_rot_z(xtc,z)  #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,y,zt[:,2:4])
        self.xt=xtc
        #print('##!! Seq: ',self.seq)
        xt=self.minimizar_x(z[:,0:2],y[:,c].T)
        return y,xt

if __name__=='__main__':
    config=ConfigICM('config_ros.yaml')
    ICM=ICM_ROS(config)
    ICM.inicializar_online()

