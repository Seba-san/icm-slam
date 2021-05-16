import numpy as np
import roslibpy # Conectarse a una red ROS
from ICM_SLAM import ICM_method



class ROS_package(ICM_method):
    def __init__(self,config):
        ICM_method.__init__(self,config)
        self.new_data=False
    
    def connect_ros(self):
        
        self.client = roslibpy.Ros(host='localhost', port=9090)
        #client.run_forever()
        self.client.run()
        if  client.is_connected:
            print('Conectado a la red ROS')
        else:
            print('No se puedo conectar a la red ROS')
            client.terminate()

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
        if self.seq<self.mediciones.size: # same amount of odometry with laser items
            z=np.array(msg['ranges'])
            z(np.isnan(z.astype('float')))=self.config.rango_laser_max # clear None value
            z=np.minimum(z+self.config.radio,z*0.0+self.config.rango_laser_max)
            self.mediciones=np.concatenate(self.mediciones,z)

    def callback_odometry(self,msg):
        """
        'twist': {'twist': {'linear': {'y':, 'x':, 'z':}, 'angular': {'y':, 'x':, 'z':}},  'covariance':, 'header': 
        'pose': {'pose': {'position': {'y':, 'x':, 'z':}, 'orientation': {'y':, 'x':, 'z':, 'w':}},'covariance':, 'child_frame_id':}
        """
        if self.seq0==0:
            self.seq0=msg['header']['seq']
        
        self.seq=msg['header']['seq']-self.seq0
        x=msg['pose']['pose']['x']
        y=msg['pose']['pose']['y']
        fi_x=msg['pose']['orientation']['x']
        fi_y=msg['pose']['orientation']['y']
        fi_z=msg['pose']['orientation']['z']
        fi_w=msg['pose']['orientation']['w']
        # Sacado de la libreria de cuaterniones
        # Fuente: https://github.com/Seba-san/pyquaternion
        t3 = +2.0 * (fi_w * fi_z + fi_x * fi_y)
        t4 = +1.0 - 2.0 * (fi_y **2 + fi_z **2)
        yaw_z = math.atan2(t3, t4)
        odo=np.array([x,y,yaw_z])
        self.odometria=np.concatenate(self.odometria,odo)
        self.new_data=True
        
    def inicializar_online(self):
        self.connect_ros()

        while True: # Solo para test, luego incorporar el servicio
            if self.new_data:
                self.new_data=False
                self.inicializar_online_process(y,xt)

        self.disconnect_ros()
        self.itererar(mapa_viejo,x)

    def inicializar_online_process(self,y,xt):
        """
        callback del servicio de ROS
        """

        #xtc=self.g(xt,u[:,t-1])  #actualizo cinemáticamente la pose
        xtc=self.odometria[:,-1]
        z=filtrar_z(self.mediciones[:,-1],self.config)  #filtro observaciones no informativas del tiempo t: [dist ang x y] x #obs
        if z.shape[0]==0:
            xt=xtc
            #x[:,t]=xt.T
            return y,xt
            #continue   #si no hay observaciones pasar al siguiente periodo de muestreo
        
        zt=tras_rot_z(xtc,z)  #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,y,zt[:,2:4])
        xt=self.minimizar_x(z[:,0:2],y[:,c].T,xt,self.odometria[:,-2:-1])

        return y,xt
