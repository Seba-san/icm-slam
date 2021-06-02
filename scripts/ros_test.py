import roslibpy
from copy import deepcopy as copy
import numpy as np


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


