"""
Esta función busca publicar en una red ROS los datos creados en un .mat 
"""

import scipy.io as sio # es para leer los datos de un .mat
import roslibpy # Para conectarse a una red ROS.
import time
import numpy as np

class ROS_bridge:
    def __init__(self):
        pass
    
    def connect_ros(self):
        
        self.client = roslibpy.Ros(host='localhost', port=9090)
        self.client.run()
        if  self.client.is_connected:
            print('Conectado a la red ROS')
        else:
            self.disconnect_ros()


        topic_laser= '/pioneer2dx/laser/scan_Lidar_horizontal'
        topic_laser_msg= 'sensor_msgs/LaserScan'
        topic_odometry= '/pioneer2dx/ground_truth/odom'
        topic_odometry_msg= 'nav_msgs/Odometry'

        self.lidar= roslibpy.Topic(self.client, topic_laser, topic_laser_msg)
        self.odometry= roslibpy.Topic(self.client, topic_odometry,
                topic_odometry_msg)

    def disconnect_ros(self):
        print('modulo desconectado de la red ROS')
        self.client.terminate()

def mat2laser_scann(medicion):
    """
    LaserScan:angle_min, range_min, scan_time, range_max, angle_increment, angle_max,ranges,
    header, intensities.
    header: stamp, frame_id,seq
    """
    D={'header':{},'angle_min': -np.pi/2,
    'angle_max': np.pi/2,
    'angle_increment': np.pi/180,
    'time_increment': 0.0,
    'scan_time': 0.0,
    'range_min': 0.5,
    'range_max': 20.0}

    D['header']={ 'seq': 1,
    'stamp':{'secs': 0,
        'nsecs': 0},
    'frame_id': "Lidar_horizontal"}
    medicion=medicion.tolist()
    D['ranges']=medicion
    D['intensities']=[]

    #import pdb; pdb.set_trace() # $3 sacar esto

    return D

def mat2odometry(odo,vel):
    """
    'twist': {'twist': {'linear': {'y':, 'x':, 'z':}, 'angular': {'y':, 'x':, 'z':}},  'covariance':, 'header': 
    'pose': {'pose': {'position': {'y':, 'x':, 'z':}, 'orientation': {'y':, 'x':, 'z':, 'w':}},'covariance':, 'child_frame_id':}
    """
    #import pdb; pdb.set_trace() # $3 sacar esto
    D={}
    D['header']={
    'frame_id': "odom_groundtruth"}
    D['child_frame_id']= "base_link"
    roll=0.0
    pitch=0.0
    yaw=odo[2]
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    D['pose']={'pose':{ 
    'position':{ 
      'x': odo[0],
      'y': odo[1],
      'z': 0.0},
    'orientation':{ 
      'x': 0.0,
      'y': 0.0,
      'z': qz,
      'w': qw}},
    'covariance': [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    
    D['twist']={'twist':{ 
    'linear':{ 
      'x': vel[0],
      'y': 0.0,
      'z': 0.0},
    'angular':{ 
      'x': 0.0,
      'y': 0.0,
      'z': vel[1]}},
    'covariance': [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    
    return D

class Header:
    def __init__(self):
        self.seq=0

    def new_message(self,D):
        """
        Actualiza secuencia y timestamp
        """
        ts=0.1
        s=self.seq*ts
        D['header']['seq']= self.seq
        D['header']['stamp']={'secs': int(s),
        'nsecs': int((s-int(s))*10**9)}
        self.seq=self.seq+1
        return D

if __name__=='__main__':
    data=sio.loadmat('../data_IJAC2018.mat')
    odometria = np.array(data['odometry'])
    z = np.array(data['observations'])
    u = np.array(data['velocities'])
    del(data)
    ros=ROS_bridge()
    ros.connect_ros()
    N=z.shape[1]
    headerLaser=Header()
    headerOdometry=Header()
    estate=0 
    #import pdb; pdb.set_trace() # $3 sacar esto
    for t in range(N):
        D=mat2laser_scann(z[:,t])
        D=headerLaser.new_message(D)
        ros.lidar.publish(roslibpy.Message(D))
        #time.sleep(0.04)
        D=mat2odometry(odometria[:,t],u[:,t])
        D=headerOdometry.new_message(D)
        ros.odometry.publish(roslibpy.Message(D))
        time.sleep(0.1)
        if float(t/N)>estate:
            print(estate*100,'%')
            estate=estate+0.1



    ros.disconnect_ros()


