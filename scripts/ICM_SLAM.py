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
from funciones_varias import *
#import logging
import sys
import math

import scipy.io as sio

import time


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

class ICM_method():
    """
    Esta clase sirve para computar los mínimos a las funciones fun_x y fun_xn. 
    Como es necesario pasar argumentos entre la función a minimizar y el minimizador, se opta por
    utilizar esta *clase* en vez de declarar variables globales. 
    """
    def __init__(self,config,x0=''):
        #ICM_external.__init__(self,config)
        self.config=config
        if x0=='':
            self.x0=np.zeros((3,1))  #guarda la pose actual (inicial en esta linea) del DDMR
        else:
            self.x0=x0

    def minimizar_xn(self,medicion_actual,mapa_visto,x_ant,x_pos,u,odometria):
        """
        xt=minimizar_xn(z[:,0:2],y[:,c].T,xt,x[:,t+1],u[:,t-1:t+1],odometria[:,t-1:t+2])
    
        Ec. (14) del paper.
        
        yy contiene las ubicaciones estimadas hasta el momento de los arboles observados una abajo de la otra,
        repitiendo observaciones repetidas e ignorando ubicaciones no observadas
        zz contiene las observaciones realizadas una abajo de la otra. La primer columna contiene
        distancias y la segunda ángulos relativos al laser del robot
         
         Entradas:
         - [distancia, ángulo] medicion_actual: Mediciones para un instante de tiempo
         - [(x,y) x c ] yy: Mapa con los landmarks vistos en un instante de
           tiempo. 
         - [x y \theta] x_ant: Pose actual estimada con este minimizador.
         - [x y \theta] x_pos: Pose futura. 
         - [v \omega] u: Señal de control del instante anterior
         - [x y \theta]  odometria: Estimación de la pose dado por la
           odometria en 2 instantes de tiempo.
        Salida:
         - x: la pose que minimiza el funcional fun_x.

        """
        self.medicion_actual=medicion_actual
        self.mapa_visto=mapa_visto
        self.u_ant_opt=u[:,0]
        self.x_ant_opt=x_ant.reshape((3,1))
        self.u_act_opt=u[:,1]
        self.x_pos_opt=x_pos.reshape((3,1))
        self.odo_opt=odometria
        x=fmin(self.fun_xn,(self.x_ant_opt+self.x_pos_opt)/2.0,xtol=0.001,disp=0)
        return x

    def fun_xn(self,x):
        """
        Función a minimizar. 

        Parte del argumento de la Ec. (14) del papaer. Esta función incluye a
        fun_x, ya que usa la información previa (fun_x) e información futura.
        En este codigo solo se incorpora la información futura. 

        Entrada:
         - x: Pose

        Salida:
         - f: Potencial energético debido al modelo cinemático y a la
           odometría.
        """
        x_pos=self.x_pos_opt
        u_act=self.u_act_opt
        odo=self.odo_opt

        f=self.fun_x(x)

        x=x.reshape((3,1))
        gg=self.g(x,u_act)-x_pos
        gg[2]=entrepi(gg[2])

        Rotador=Rota(x[2][0])
        ooo=np.zeros((3,1))
        # Calcula la diferencia entre los vectores desplazamiento. Entre la
        # odometria y la pose x (de forma relativa, con respecto a la pose anterior)
        # Ec. (16) del paper.
        ooo[0:2]=np.matmul(Rota(odo[2,1]),(odo[0:2,2]-odo[0:2,1]).reshape((2,1)))\
                -np.matmul(Rotador,x_pos[0:2]-x[0:2])

        ooo[2]=odo[2,2]-odo[2,1]-x_pos[2]+x[2]
        ooo[2]=entrepi(ooo[2])

        f=np.matmul(np.matmul(gg.T,self.config.R),gg)\
           +self.config.cte_odom*np.matmul(ooo.T,ooo)\
           +f
        return f

    def minimizar_x(self,medicion_actual,mapa_visto):
        """
        x=minimizar_x(self,medicion_actual,mapa_visto,x_ant,u_ant,odometria):

        Ec. (11) del paper. 

        Entradas:
         - [distancia, ángulo] medicion_actual: Mediciones para un instante de
           tiempo, filtradas (sin outliers).
         - [(x,y) x c ] mapa_visto: Mapa con los landmarks vistos en un instante de
           tiempo.
         - [x y \theta] x_ant: Pose anterior estimada con este minimizador.
         - [v \omega] u_ant: Señal de control del instante anterior
         - [x y \theta]  odometria: Estimación de la pose dado por la
           odometria en 2 instantes de tiempo.
        Salida:
         - x: la pose que minimiza el funcional fun_x. 
        """
        self.medicion_actual=medicion_actual
        self.mapa_visto=mapa_visto
        x=fmin(self.fun_x,self.xt,xtol=0.001,disp=0)
        return x

    def fun_x(self,x):
        """
        Argumento de la Ec. (11) del paper.

        Función a minimizar que depende de:
         - Odometria
         - Estimacion de pose anterior
         - Modelo cinemático de movimiento
         - Modelo de las observaciones
         - Observaciones en un instate dado
        
        Parámetros
        -----------
        
        Entrada:
         - x: posición.

        Salida:
         - f: potencial energético.

        """

        z=self.medicion_actual
        #x_ant=self.x_ant_opt# $2 ver
        x_ant=self.xt
        #u_ant=self.u_ant_opt# $2 ver
        #print(self.odometria)
        odo=self.odometria[:,-2:] # Los ultimos 2
        #print(odo)
        # vector desplazamiento entre las estimacion de pose anterior y la pose
        # actual X.
        #gg=x.reshape((3,1))-self.g(x_ant,u_ant)
        #gg[2]=entrepi(gg[2])

        hh=self.h(x,z)
        Rotador=Rota(x_ant[2][0])
        ooo=np.zeros((3,1))
        # Calcula la diferencia entre los vectores desplazamiento. Entre la
        # odometria y la pose x (de forma relativa, con respecto a la pose anterior)
        # Ec. (16) del paper.
        ooo[0:2]=np.matmul(Rota(odo[2,0]),(odo[0:2,1]-odo[0:2,0]).reshape((2,1)))\
                -np.matmul(Rotador,x[0:2].reshape((2,1))-x_ant[0:2])
        
        ooo[2]=odo[2,1]-odo[2,0]-x[2]+x_ant[2]
        ooo[2]=entrepi(ooo[2])
        
        """
        f=np.matmul(np.matmul(gg.T,self.config.R),gg)+\
           hh+\
           self.config.cte_odom*np.matmul(ooo.T,ooo)
        return f
        """

        f=hh+self.config.cte_odom*np.matmul(ooo.T,ooo)
        return f

    def load_data(self,mapa_obj,mediciones,u,odometria,x0=''):
        """
        Antes de iterar es necesario cargar todas las observaciones y las
        acciones de control.
        """
        self.mediciones=mediciones
        self.u=u
        self.mapa_obj=copy(mapa_obj)
        self.odometria=odometria
        #print('Cargando datos')
        #print(self.mapa_obj.landmarks_actuales)

        if x0=='':
            self.x0=np.zeros((3,1))  #guarda la pose actual (inicial en esta linea) del DDMR
        else:
            self.x0=x0

    def inicializar(self,x):
        """
        Este método inicializa las variables para que luego se pueda refinar mediante ICM el mapa y las poses históricas del
        vehículo.

        Esta función es la que se implementa online. Toma los datos de las
        mediciones, odometria, estados anteriores, etc. para estimar la pose
        actual.

        Argumentos de entrada:
        ----------------------
        Variables:
         - Posiciones actuales 'x'
        Constantes:
         - 'config'
         - Mediciones 'z'
         - acciones de control 'u'
        
        Argumentos de salida:
        ---------------------
         - Mapa actualizado 'mapa_refinado'
         - Posiciones refinadas 'x' 

         Como referencia toma como maximo 0.01 segundos por iteracion. (no sé
         de que depende.)
        """

        xt=copy(self.x0)
        y=np.zeros((2,self.config.L)) #guarda la posicion de los a lo sumo L arboles del entorno
        self.mapa_obj.clear_obs()
        z=filtrar_z(self.mediciones[:,0],self.config)  #filtro la primer observacion [dist ang x y] x #obs
        u=self.u
        odometria=self.odometria

        if z.shape[0]==0:
            #print('skip!!!!!!!!!')
            return y,x
            #continue #si no hay observaciones pasar al siguiente periodo de muestreo
        
        zt=tras_rot_z(xt,z) #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,y,zt[:,2:4])
    
        #BUCLE TEMPORAL
        for t in range(1,self.config.Tf):
            # aca va el bucle ros.
            #inicio=time.time()
            xtc=self.g(xt,u[:,t-1])  #actualizo cinemáticamente la pose
            z=filtrar_z(self.mediciones[:,t],self.config)  #filtro observaciones no informativas del tiempo t: [dist ang x y] x #obs
            if z.shape[0]==0:
                xt=xtc
                x[:,t]=xt.T
                continue   #si no hay observaciones pasar al siguiente periodo de muestreo
            
            zt=tras_rot_z(xtc,z)  #rota y traslada las observaciones de acuerdo a la pose actual
            y,c=self.mapa_obj.actualizar(y,y,zt[:,2:4])
            xt=self.minimizar_x(z[:,0:2],y[:,c].T,xt,u[:,t-1],odometria[:,t-1:t+1])
            x[:,t]=xt
            #final=time.time()
            #print('Tiempo transcurrido: ',inicio-final)
        
        #filtro ubicaciones estimadas

        yy=self.mapa_obj.filtrar(y)
        yy=yy[:,:self.mapa_obj.landmarks_actuales]
        mapa_inicial=copy(yy)

        #graficar(x,yy,iteracionICM)#gráficos
        return mapa_inicial,x 


    def itererar(self,mapa_viejo,x):
        """
        Este método refina mediante ICM el mapa y las poses históricas del
        vehículo.

        Argumentos de entrada:
        ----------------------
        Variables:
         - Mapa a refinar 'mapa_viejo'
         - Posiciones actuales 'x'
        Constantes:
         - 'config'
         - Mediciones 'z'
         - acciones de control 'u'
        
        Argumentos de salida:
        ---------------------
         - Mapa actualizado 'mapa_refinado'
         - Posiciones refinadas 'x' 
        """

        xt=copy(self.x0)
        y=np.zeros((2,self.config.L)) #guarda la posicion de los a lo sumo L arboles del entorno
        self.mapa_obj.clear_obs()
        z=filtrar_z(self.mediciones[:,0],self.config)  #filtro la primer observacion [dist ang x y] x #obs
        odometria=self.odometria
        u=self.u
        if z.shape[0]==0:
            #print('skip!!!!!!!!!')
            return mapa_viejo,x
            #continue #si no hay observaciones pasar al siguiente periodo de muestreo
        
        zt=tras_rot_z(xt,z) #rota y traslada las observaciones de acuerdo a la pose actual
        y,c=self.mapa_obj.actualizar(y,mapa_viejo,zt[:,2:4])
    
        #BUCLE TEMPORAL
        for t in range(1,self.config.Tf):
            z=filtrar_z(self.mediciones[:,t],self.config)  #filtro observaciones no informativas del tiempo t: [dist ang x y] x #obs
            if z.shape[0]==0:
                xt=(xt.reshape(3)+x[:,t+1])/2.0
                x[:,t]=xt
                continue #si no hay observaciones pasar al siguiente periodo de muestreo
            
            zt=tras_rot_z(x[:,t],z)  #rota y traslada las observaciones de acuerdo a la pose actual
            y,c=self.mapa_obj.actualizar(y,mapa_viejo,zt[:,2:4])
            if t+1<self.config.Tf:
                xt=self.minimizar_xn(z[:,0:2],y[:,c].T,xt,x[:,t+1],u[:,t-1:t+1],odometria[:,t-1:t+2])
            else:
                xt=self.minimizar_x(z[:,0:2],y[:,c].T,xt,u[:,t-1],odometria[:,t-1:t+1])
            x[:,t]=xt
        
        #filtro ubicaciones estimadas

        yy=self.mapa_obj.filtrar(y)
        yy=yy[:,:self.mapa_obj.landmarks_actuales]
        #print(mapa.Landmarks_actuales)
        #print(mapa.cant_obs_i)
        #print(yy)
        
      
        mapa_refinado=copy(yy)

        #graficar(x,yy,iteracionICM)#gráficos
        return mapa_refinado,x
    

    def g(self, xt,ut):
        """
        Modelo de la odometría
        ========================
        Actualización de la pose a partir de las señales de control usando solo la
        cinemática del vehículo. Como referencia revisar definicion en el paper,
        página 8/15. 
        Entradas:
         - :math:`x_t=[ x_{t,x}, x_{t,y}, x_{t,\theta}]^T` 
         - :math:`u_t=[ v_t, \omega_t ]^T`
         - config: Objeto que contiene todas las configuraciones
    
        Salida
         - :math:`x_{t+1}=[ x_{t+1,x}, x_{t+1,y}, x_{t+1,\theta}]^T`
    
         A futuro poner la ecuación en este lugar.
        """
    
        xt=xt.reshape((3,1))
        ut=ut.reshape((2,1))
        S=np.array([[(np.cos(xt[2]))[0],0.0],
            [np.sin(xt[2])[0],0.0],
            [0.0,1.0]])
        gg=xt+self.config.deltat*np.matmul(S,ut).reshape((3,1))
        return gg
    
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
        'Externas'
         - [2 x Lact] Mapa: Es una matriz con las posiciones 2D de todos árboles.
         - yy o mapa_referencia: Es el mapa de referencia, que NO se modifca.
         - [(x,y) x Nobs] obs: Lista de observaciones en coordenadas cartecianas.
           Nobs son la cantidad de observaciones filtradas (sin outliers). 
         'Internas'
         - [int] Lact: cantidad de árboles hasta el momento (Landmarks
           activos/actuales)

        Salidas: mapa,c
        'Externas'
         - [2 x Lact] Mapa: Es una matriz con las posiciones 2D de 'Lact' árboles.
         - vector de etiquetas 'c': Vector de etiquetas de cada landmark. Dice a que landmark
           corresponde cada medición.
        'Internas'
         - int [1 x Lact] cant_obs_i: Conteo de la cantidad de veces que se observo
           un árbol.
         - [int] Lact: cantidad de árboles actualizado.

        Mapa es la estimación de los árboles que se tiene hasta el momento dentro de la iteracion ICM.
        **yy** es la estimación de los árboles que se usará para realizar el etiquetado de las obs nuevas. 
        En la iteracion 0 yy=mapa, pero en las iteraciones siguientes yy es el mapa final estimado en la iteración ICM
        anterior.
        
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
                    #mapa[:,i]=np.sum(zt[c==i,2:4],axis=0)/(cant_obs_i[i]+len(c[c==i]))\
                    #        +mapa[:,i]*cant_obs_i[i]/(cant_obs_i[i]+len(c[c==i]))
    
                    #                obs o zt?
                    mapa[:,i]=np.sum(obs[c==i],axis=0)/(cant_obs_i[i]+len(c[c==i]))\
                            +mapa[:,i]*cant_obs_i[i]/(cant_obs_i[i]+len(c[c==i]))
                    
                    cant_obs_i[i]=cant_obs_i[i]+len(c[c==i])

        self.landmarks_actuales=Lact
        self.cant_obs_i=cant_obs_i
    
        return mapa,c


    def filtrar(self,mapa):
       """
       [y,cant_obs_i,Lact]=filtrar_y(y,cant_obs_i)
       
       Se filtra el mapa, eliminando landmarks poco observados y unificando
       landmarks cercanos. 
       
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

