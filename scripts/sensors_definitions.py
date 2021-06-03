import numpy as np
from ICM_SLAM import Sensor
from copy import deepcopy as copy
import math

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
