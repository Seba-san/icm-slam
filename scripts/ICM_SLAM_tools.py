#Importa todos los paquetes
import numpy as np
import yaml
import numpy.matlib
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.signal import medfilt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from copy import deepcopy as copy
#import logging
import sys
import math

import scipy.io as sio

import roslibpy # Conectarse a una red ROS
import time
import matplotlib.pyplot as plt


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

class ConfigICM:
    """
    Núcleo de todos los parámetros de configuración y constantes. 
    """
    def __init__(self,configFile='config_default.yaml',D={}):
        if not D:
            arch=open(configFile, 'r')
            Data = yaml.load(arch,Loader=yaml.FullLoader)
            D=Data['D']
            

        # parámetros por default
        self.N=D['N']  #cantidad de iteraciones de ICM
        self.deltat=D['deltat']  #periodo de muestreo
        self.L=D['L']  #cota superior de la cantidad de objetos (max landmarks)
        
        self.Q=np.eye(2)#matriz de covarianza de las observaciones
        self.Q[0,0]=D['Q'][0] 
        self.Q[1,1]=D['Q'][1] 
        
        self.R=np.eye(3) #matriz de covarianza del motion model
        self.R[0,0]=D['R'][0]
        self.R[1,1]=D['R'][1]
        self.R[2,2]=D['R'][2]

        self.cte_odom=D['cte_odom']  #S=diag([cte_odom,cte_odom,cte_odom]) matriz de peso de los datos odométricos
        self.cota=D['cota']  #cantidad de veces que hay q ver un arbol para q se considere un arbol
        self.dist_thr=D['dist_thr']  #distancia máxima para que dos obs sean consideradas del mismo objeto
        self.dist_thr_obs=D['dist_thr_obs']  #distancia máxima para que dos obs sean consideradas del mismo objeto en el proceso de filtrado de las observaciones
        self.rango_laser_max=D['rango_laser_max']  #alcance máximo del laser
        self.radio=D['radio'] #radio promedio de los árboles

        self.topic_laser=D['topic_laser']
        self.topic_laser_msg=D['topic_laser_msg']
        
        self.topic_odometry=D['topic_odometry']
        self.topic_odometry_msg=D['topic_odometry_msg']

        self.file=D['file']
        self.time=D['time'] # tiempo de test para la inicializacion online.

    def set_Tf(self,Tf):
        self.Tf=Tf  #cantidad total de periodos de muestreo

class Mapa:
    """
    Esta clase define los metodos utilizados para mapear y refinar un mapa.
    """

    def __init__(self,config):
        # Parámetros de configuracion inicial
        self.L=config.L
        self.cota=config.cota
        self.dist_thr=config.dist_thr
        
        self.landmarks_actuales=0
        self.clear_obs() 
        #self.mapa=np.zeros((2,config.L)) #guarda la posicion de los a lo sumo L arboles del entorno
    
    def clear_obs(self):
        """
        Inicialización de variables internas

        self.cant_obs_i=np.zeros(self.L)  #guarda la cantidad de veces que se observó el i-ésimo árbol

        """
        self.cant_obs_i=np.zeros(self.L)  #guarda la cantidad de veces que se observó el i-ésimo árbol

    def actualizar(self,mapa,mapa_referencia,obs):
        """
        actualizar(self,mapa,mapa_referencia,obs)
        
        Actualiza las variables relacionadas con la construcción del mapa.
        Como argumentos de entrada son el mismo mapa y las observaciones de una sola
        medicion.
    
        Parámetros:
        -----------
    
        Entradas: mapa, mapa_referencia, obs
         - [2 x Lact] Mapa: Es una matriz con las posiciones 2D de todos árboles.
         - yy o mapa_referencia: Es el mapa de referencia, que NO se modifca.
         - [(x,y) x Nobs] obs: Lista de observaciones en coordenadas cartecianas.
           Nobs son la cantidad de observaciones filtradas (sin outliers). 

        Salidas: mapa,c
        'Externas'
         - [2 x Lact] Mapa: Es una matriz con las posiciones 2D de 'Lact' árboles.
         - vector de etiquetas 'c': Vector de etiquetas de cada landmark. Dice a que landmark
           corresponde cada medición.
        'Internas'
         - int [1 x Lact] cant_obs_i: Conteo de la cantidad de veces que se observo
           un árbol.
         - [int] Lact: cantidad de árboles actualizado.

        #actualizo (o creo) ubicación de arboles observados
        """
        Lact=self.landmarks_actuales
        cant_obs_i=self.cant_obs_i

        if Lact==0:#este bucle es solamente para t=0 de la iteración ICM 0
            c=fcluster(linkage(pdist(obs)),self.dist_thr)-1  #calculo clusters iniciales
            Lact=np.max(c)+1  #cantidad de arboles iniciales
            for i in range(Lact):
                mapa[:,i]=np.mean(obs[c==i,:],axis=0).T #calculo el centro de cada cluster
                cant_obs_i[i]=len(c[c==i])
    
        else:
            #me fijo si las observaciones corresponden a un arbol existente
            distancias=cdist(mapa_referencia[:,:Lact].T,obs)#matriz de distancias entre yy y las obs nuevas
            min_dist=np.amin(distancias,axis=0)#distancia minima desde cada obs a un arbol de yy
            c=np.argmin(distancias,axis=0)#etiqueta del arbol de yy que minimiza la distancia a cada obs nueva
            c[min_dist>self.dist_thr]=-1#si esta lejos de los arboles de yy le asigno la etiqueta -1 momentaneamente
            #armo cluster con observaciones de arboles nuevos
            ztt=obs[min_dist>self.dist_thr,:]#extraigo las obs nuevas que estan lejos de los árboles de yy
            if ztt.shape[0]>1:#si hay mas de una observacion de un arbol no mapeado aun
                cc=Lact+fcluster(linkage(pdist(ztt[:,2:4])),self.dist_thr)-1 #calculo clusters y le coloco una etiqueta nueva (a partir de Lact=max etiqueta+1)
                c[c==-1]=cc#a todos los árboles con etiqueta -1 le asigno su nueva etiqueta
    
            elif ztt.shape[0]==1:#si hay sólo una observacion de un arbol no mapeado aun
                c[c==-1]=Lact
    
            Lact=np.amax(np.append(c+1,Lact))#actualizo la cantidad de arboles mapeados hasta el momento
            #actualizo (o creo) ubicación de arboles observados
            for i in range(Lact):
                if len(c[c==i])>0:

                    #import pdb; pdb.set_trace() # $3 sacar esto
                    #mapa[:,i]=np.sum(zt[c==i,2:4],axis=0)/(cant_obs_i[i]+len(c[c==i]))\
                    #        +mapa[:,i]*cant_obs_i[i]/(cant_obs_i[i]+len(c[c==i]))
                    
                    mapa[:,i]=np.sum(obs[c==i],axis=0)/(cant_obs_i[i]+len(c[c==i]))\
                            +mapa[:,i]*cant_obs_i[i]/(cant_obs_i[i]+len(c[c==i]))
                    
                    cant_obs_i[i]=cant_obs_i[i]+len(c[c==i])

        self.landmarks_actuales=Lact
        self.cant_obs_i=cant_obs_i
        #print('Lact: ',Lact)
        #print('Cant_obs: ',cant_obs_i[:Lact])
    
        return mapa,c


    def filtrar(self,mapa):
       """
       [y,cant_obs_i,Lact]=filtrar_y(y,cant_obs_i)
       
       Se filtra el mapa, eliminando landmarks poco observados y unificando
       landmarks cercanos. 
     #  
       Parámetros
       ----------
    
       Entradas: mapa
       'Externas'
        - y o mapa: Mapa de entrada
       
       'Internas'
        - cant_obs_i: Cantidad de veces que se observo cada árbol ordenados por su
          índice
    
       Salidas: mapa_filtrado
       'Externas'
        - yy o mapa_filtrado: mapa filtrado con los árboles más observados
       'Internas'
        - cant_obs_i: Cantidad de observaciones luego de filtrar 
       """
       Lact=self.landmarks_actuales 
       cant_obs_i=self.cant_obs_i

       cant_obs_i=cant_obs_i[0:Lact] #saco ceros innecesarios
       ind=np.where(cant_obs_i<self.cota)[0]  #indices de arboles poco observados
       if ind.size>0:  #si hay arboles poco observados
           Lact=Lact-ind.size  #reduzco la cantidad de arboles observados hasta el momento
           # Cambiar esto, poner la media en vez de una cota estática.
           ind2=np.where(cant_obs_i>=self.cota)[0] #indices de arboles observados muchas veces
           mapa=mapa[:,ind2] #elimino las posiciones estimadas de los arboles vistos pocas veces
           np.concatenate((mapa,np.zeros((2,len(ind)))),axis=1)  #le devuelvo a y su dimension original completando con ceros
           cant_obs_i=cant_obs_i[ind2] #elimino las cantidades observadas de los arboles vistos pocas veces
    
       a=squareform(pdist(mapa[:,0:Lact].T))  #calculo la matriz de distancias 2a2 de todas las posiciones de arboles observados
       a[a==0]=np.amax(a) #reemplazo los ceros (de la diagonal generalmente) para que no interfiera en el calculo de minimos en las siguientes lineas
       b=np.argmin(a,axis=0) #vector que contiene contiene el valor j en la entrada i, si el arbol j es el más cercano al arbol i
       a=np.amin(a,axis=0) #vector que contiene la distancia minima entre los arboles i y j de la linea anterior
       ind=np.where(a<self.dist_thr)[0] #indices donde la distancia entre dos arboles es muy chica
       c=np.arange(Lact)  #contiene los indices de los arboles
       if ind.size>0:  #si hay arboles muy cercanos los unifico
           for i in range(len(ind)): #el arbol ind[i] tiene al arbol b[ind[i]] muy cercano
               c[c==c[b[ind[i]]]]=c[ind[i]]  #le asigno al arbol b[ind[i]] (y a todos los cercanos a él) el indice del arbol ind[i]
    
       for i in range(Lact-1,-1,-1):
           if len(c[c==i])==0:  #si el arbol i perdió su indice por ser cercano a uno de indice menor
               c[c>=i]=c[c>=i]-1 #a todos los de indice mayor a i le resto 1... ya que el indice i ya no existe
    
       Lact=max(c)+1 #actualizo la cantidad de arboles observados luego del filtro
       mapa_filtrado=np.zeros((2,self.L)) #contendrá la posición media ponderada de acuerdo a cant_obs_i entre todos los arboles unificados por estar cercanos 
       cant_obs=np.zeros(self.L)  #reemplazará a cant_obs_i
       for i in range(Lact):
           cant_obs[i]=np.sum(cant_obs_i[c==i])
           mapa_filtrado[:,i]=np.sum(mapa[:,c==i]*np.matlib.repmat(cant_obs_i[c==i],2,1),axis=1)/cant_obs[i] #calculo el centro de cada nuevo cluster
       
       self.landmarks_actuales=Lact
       self.cant_obs_i=cant_obs

       return mapa_filtrado

class ROS:
    def __init__(self):
        pass

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
        service = roslibpy.Service(self.client, '/icm_slam/iterative_flag','std_srvs/SetBool')
        service.advertise(self.icm_iterations_service)
         
    def disconnect_ros(self):
        print('modulo desconectado de la red ROS')
        self.client.terminate()

    def icm_iterations_service(self,request,response):
        """
        request['data']='start'
        """
        response['success']=True
        response['message']='Working...'
        self.iterations_flag=True
        return True
    
    def principal_callback(self):
        """
        A medida que esten todos los datos disponibles, va agregando al array
        de los sensores los datos de forma ordenada.
        """
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

class Sensor:
    """
    Esta clase esta en construccion, aún no se usa ni funciona.
    """
    def __init__(self,config='',name='name',topic='',topic_msg='',principalCallback=''):
        self.msgs=[]
        self.value=np.array([])
        self.k0=0
        self.config=config
        self.name=name
        self.topic=topic
        self.topic_msg=topic_msg
        self.principalCallback=principalCallback
        self.c=0
        #self.estructura=estructura # Ver como incorporarlo
    
    def callback(self,msg):
        D=self.header_process(msg)
        """
        Poner codigo de lectura aquí.
        """
        print('Hay que modificar esta funcion')
        self.msgs.append(copy(D))
        self.principal_callback()

    def principal_callback(self):
        print("a este principal no...")
        pass

    def sort(self,k):
        """
        Busca dentro del historial de mensajes la secuencia correspondiente.
        Para agilizar la busqueda, si el valor inicial propuesto, coincide,
        rapidamente devuelve el indice.
        Ver si se puede hacer con el timestamp en lugar de hacerlo con el
        numero de secuencia.
        """
        ts=self.config.deltat
        now=k*ts+self.t0
        # Calculo aproximado
        """
        if k!=1:
            k1=round(self.k0/(k-1)) # ver la convergencia de esta serie..
        else:
            k1=k
        """
        if self.c!=0:
            k1=int(np.ceil(self.k0*k/self.c))
            if len(self.msgs)-1<k1:
                k1=len(self.msgs)-1
        else:
            k1=k


        if abs(self.msgs[k1]['stamp']-now)<ts:
            self.k0=k1
            self.c=k
            #return k1*k
            return self.msgs[k1]['data'],True

        else:

            L=len(self.msgs)
            for i in range(self.k0,L):
                 if abs(self.msgs[i]['stamp']-now)<ts:
                     print('Warning 0: datos desincronizados, adaptando...')
                     print('buscando: ',k1)
                     print('k1: ',k1, 'k: ',k,'k0: ',self.k0,'conv: ',self.k0/self.c)
                     print('diferencia: ',i-self.k0)
                     self.k0=i
                     self.c=k
                     #return i
                     print('Encontrado:', i)
                     return self.msgs[i]['data'],True

            print('Warning 1: no se encuentra la secuencia buscada')
            print('Sensor: ',self.name)
            #print('mensaje: ',self.msgs[k])
            print('iteracion: ',k)
            print('tiempo actual: ',now)
            #sys.exit()
            #self.k0=len(self.msgs)-1
            #self.c=k
            return None,False

    def header_process(self,msg):
        """
        uint32 seq
        time stamp
        string frame_id
        """
        seq=msg['header']['seq']
        stamp=msg['header']['stamp']['secs']+msg['header']['stamp']['nsecs']*10**(-9)
        D={}
        D['seq']=seq
        D['stamp']=stamp

        return D

    def subscribe(self,client):

        listener = roslibpy.Topic(client, self.topic,
                self.topic_msg)
        listener.subscribe(self.callback)

    def set_t0(self):
        self.t0=self.msgs[0]['stamp']

"""
Funciones Varias
"""

def entrepi(angulo):
    """
    retorna el ángulo equivalente entre -pi y pi
    """
    angulo=np.mod(angulo,2*np.pi)
    if angulo>np.pi:
      angulo=angulo-2*np.pi
    
    return angulo

def tras_rot_z(x,z):
    """
     - Rota y traslada las observaciones de acuerdo a la pose actual
     - Transforma del body frame al global frame

    :math:`x_t=[ x_{t,x}, x_{t,y}, x_{t,\theta}]^T` 
    :math:`z_i=\{z_{t,i}:i=1,\cdots,n_t\}` :math:`z_{t,i}=[ z_{t,i,d}, z_{t,i,\theta} ]^T`

    """

    x=x.reshape((3,1))
    ct=np.cos(x[2]-np.pi/2.0)[0]
    st=np.sin(x[2]-np.pi/2.0)[0]
    R=np.array([[ct,st],[-st,ct]])
    z[:,2:4]=np.matmul(z[:,2:4],R)+np.matlib.repmat(x[0:2].T,z.shape[0],1)
    return z

def Rota(theta):
    """
    Arma la matriz de rotación en 2D a partir de un ángulo en *radianes*.
    """
    A=np.array([[np.cos(theta),np.sin(theta)],
        [-np.sin(theta),np.cos(theta)]])
    return A

def calc_cambio(y,mapa_viejo):
    min_dist=np.amin(cdist(mapa_viejo.T,y.T),axis=0)
    cambio_minimo=np.amin(min_dist)
    cambio_maximo=np.amax(min_dist)
    cambio_medio=np.mean(min_dist)
    return cambio_minimo,cambio_maximo,cambio_medio

def graficar(x,yy,odometria,N=0):
    plt.figure(N) 
    plt.plot(x[0],x[1], 'b')
    plt.plot(odometria[0],odometria[1], 'g')
    plt.plot(yy[0],yy[1], 'b*')
    plt.axis('equal')
    #plt.pause(0.1)
    plt.show()

class graficar2:
    def __init__(self):
        pass
    
    def data(self,x,yy,odometria,N=0):
        plt.figure(N)
        plt.clf()
        plt.plot(x[0],x[1], 'b')
        plt.plot(odometria[0],odometria[1], 'g')
        plt.plot(yy[0],yy[1], 'b*')
        plt.axis('equal')
        plt.pause(0.01)

    def show(self):
        print('Cerrar img para continuar')
        plt.show()

def graficar_cambio(cambios_minimos,cambios_maximos,cambios_medios):
    plt.figure(100) 
    plt.plot(cambios_minimos, 'b--')
    plt.plot(cambios_maximos, 'b--')
    plt.plot(cambios_medios, 'b')
    plt.show()

