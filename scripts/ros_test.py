import roslibpy
from copy import deepcopy as copy
import numpy as np


class Sensor:
    """
    Esta clase esta en construccion, aún no se usa ni funciona.
    """
    def __init__(self,config='',name='name',topic='',topic_msg=''):
        self.msgs=[]
        self.value=np.array([])
        self.k0=0
        self.config=config
        self.name=name
        self.topic=topic
        self.topic_msg=topic_msg
        #self.estructura=estructura # Ver como incorporarlo

    def callback(self,msg):
        D=self.header_process(msg)
        """
        Poner codigo de lectura aquí.
        """
        self.msgs.append(copy(D))
        self.principal_callback()

    def principal_callback(self):
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
        k1=round(self.k0/(k-1)) # ver la convergencia de esta serie..

        if abs(self.msgs[k1*k]['stamp']-now)<ts:
            self.k0=k1*k
            return k1*k
        else:
            print('Warning 0: datos desincronizados, adaptando')
            L=len(self.msgs)
            for i in range(self.k0,L):
                 if abs(self.msgs[i]['stamp']-now)<ts:
                     print('diferencia: ',i-self.k0)
                     self.k0=i
                     return i

            print('Sensor Error: ',self.name)
            #print('mensaje: ',self.msgs[k])
            print('k: ',k)
            print('t0: ',self.t0)
            print('now: ',now)
            print('Error 0: no se encuentra la secuencia buscada')
            #sys.exit()
            return
        pass

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

