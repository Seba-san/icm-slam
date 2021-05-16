import numpy as np
import roslibpy # Conectarse a una red ROS

def filtrar_z(z,config):
    """
    zz=filtrar_z(z,config)

    #Elimina observaciones aisladas o de rango máximo.
    #Salida **zz** : es una matriz de 2 columnas que alista una abajo de otra las distancias y los angulos en los cuales hay una observación "positiva".

    Entradas:
     - [float]_181x1 z: Medición del lidar en un instante de tiempo. De -90 a
       90 (ejemplo)
     - config: parámetros de configuración

    Salidas:
     - numpy array [dist ang x y] zz: Salida filtrada y reformateada, $1 revisar!!   
    """
    z=medfilt(z)  #filtro de mediana con ventana 3 para borrar observaciones laser outliers
    zz=copy(z) #copia para no sobreescribir
    #hallo direcciones con observacion, el [0] es para solo quedarte con el
    #array, no con la tupla
    nind=np.where(z<config.rango_laser_max)[0] 
    if len(nind)>1:
      z=z[nind] #solo me quedo con las direcciones observadas
      z=np.concatenate((np.cos(nind*np.pi/180.0)*z,
                        np.sin(nind*np.pi/180.0)*z)).reshape((len(nind),2),order='F') #ahora z tiene puntos 2D con la ubicacion relativa de las observaciones realizadas
      c=squareform(pdist(z))  #matriz de distrancia entre obs
      #modifico la diagonal con un numero grande
      c[c==0]=100 #$1 ojo, esto depende del rango máximo
      c=np.amin(c,axis=0) #calculo la distancia al objeto más cercano de cada observacion
      nind=nind[c<=config.dist_thr] #elimino direcciones aisladas
      zz=np.concatenate((zz[nind],nind*np.pi/180.0)).reshape((len(nind),2),order='F') #ahora zz contiene las distancias y la direccion (en radianes) de las observaciones no aisladas
      zzz=np.concatenate((zz[:,0],zz[:,0])).reshape((len(nind),2),order='F')\
              *np.concatenate((np.cos(zz[:,1]),np.sin(zz[:,1]))).reshape((len(nind),2),order='F') #contiene la posicion relativa de las observaciones no aisladas
      zz=np.concatenate((zz,zzz),axis=1)
    else:
      zz=np.array([])

    return zz


class ROS_package:
    def __init__(self):
        pass
    
    def connect_ros(self):

        client = roslibpy.Ros(host='localhost', port=9090)
        client.run()
        if  client.is_connected:
            print('Conectado a la red ROS')
        else:
            print('No se puedo conectar a la red ROS')
            client.terminate()

        listener_laser = roslibpy.Topic(client, self.config.topic_laser,
                self.config.topic_laser_msg)
        listener_laser.subscribe(self.callback_laser)

        listener_odometry = roslibpy.Topic(client, self.config.topic_odometry,
                self.config.topic_odometry_msg)
        listener_odometry.subscribe(self.callback_odometry)
        
        try:
               while True:
                   pass
        except KeyboardInterrupt:
               client.terminate()
 
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
        self.inicializar_online()
        
