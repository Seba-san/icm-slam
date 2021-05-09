
def filtrar_y(y,cant_obs_i,Lact,config):
    """
    [y,cant_obs_i,Lact]=filtrar_y(y,cant_obs_i,Lact,config)
    
    Se filtra el mapa, eliminando landmarks poco observados y unificando
    landmarks cercanos. 
    
    Parámetros
    ----------

    Entrada:
     - y: Mapa de entrada
     - cant_obs_i: Cantidad de veces que se observo cada árbol ordenados por su
       índice
     - Lact: Cantidad de árboles vistos hasta ahora
     - config: Objeto que contiene todos los parámetros de configuración

    Salida:
     - yy: mapa filtrado con los árboles más observados
     - cant_obs: Cantidad de observaciones luego de filtrar 
     - Lact: Cantidad de árboles vistos actualizado


    """

    cant_obs_i=cant_obs_i[0:Lact] #saco ceros innecesarios
    ind=np.where(cant_obs_i<config.cota)[0]  #indices de arboles poco observados
    if ind.size>0:  #si hay arboles poco observados
        Lact=Lact-ind.size  #reduzco la cantidad de arboles observados hasta el momento
        ind2=np.where(cant_obs_i>=config.cota)[0] #indices de arboles observados muchas veces
        y=y[:,ind2] #elimino las posiciones estimadas de los arboles vistos pocas veces
        np.concatenate((y,np.zeros((2,len(ind)))),axis=1)  #le devuelvo a y su dimension original completando con ceros
        cant_obs_i=cant_obs_i[ind2] #elimino las cantidades observadas de los arboles vistos pocas veces

    a=squareform(pdist(y[:,0:Lact].T))  #calculo la matriz de distancias 2a2 de todas las posiciones de arboles observados
    a[a==0]=np.amax(a) #reemplazo los ceros (de la diagonal generalmente) para que no interfiera en el calculo de minimos en las siguientes lineas
    b=np.argmin(a,axis=0) #vector que contiene contiene el valor j en la entrada i, si el arbol j es el más cercano al arbol i
    a=np.amin(a,axis=0) #vector que contiene la distancia minima entre los arboles i y j de la linea anterior
    ind=np.where(a<config.dist_thr)[0] #indices donde la distancia entre dos arboles es muy chica
    c=np.arange(Lact)  #contiene los indices de los arboles
    if ind.size>0:  #si hay arboles muy cercanos los unifico
        for i in range(len(ind)): #el arbol ind[i] tiene al arbol b[ind[i]] muy cercano
            c[c==c[b[ind[i]]]]=c[ind[i]]  #le asigno al arbol b[ind[i]] (y a todos los cercanos a él) el indice del arbol ind[i]

    for i in range(Lact-1,-1,-1):
        if len(c[c==i])==0:  #si el arbol i perdió su indice por ser cercano a uno de indice menor
            c[c>=i]=c[c>=i]-1 #a todos los de indice mayor a i le resto 1... ya que el indice i ya no existe

    Lact=max(c)+1 #actualizo la cantidad de arboles observados luego del filtro
    yy=np.zeros((2,config.L)) #contendrá la posición media ponderada de acuerdo a cant_obs_i entre todos los arboles unificados por estar cercanos 
    cant_obs=np.zeros(config.L)  #reemplazará a cant_obs_i
    for i in range(Lact):
        cant_obs[i]=np.sum(cant_obs_i[c==i])
        yy[:,i]=np.sum(y[:,c==i]*np.matlib.repmat(cant_obs_i[c==i],2,1),axis=1)/cant_obs[i] #calculo el centro de cada nuevo cluster

    return yy,cant_obs,Lact

def actualizar_mapa(mapa,yy,obs,Lact,config):
    """
    mapa,cant_obs_i,c,Lact=actualizar_mapa(mapa,yy,obs,Lact,config)
    
    Actualiza las variables relacionadas con la construcción del mapa. Como
    argumentos de entrada son el mismo mapa y las observaciones (de una sola
    medicion? o pueden ser varias observaciones?). 
    Ver una forma de hacer una clase "Mapa" para hacer estas actualizaciones,
    ya que hay mucha información dedundante. 

    Mapa es la estimación de los árboles que se tiene hasta el momento dentro de la iteracion ICM.
    **yy** es la estimación de los árboles que se usará para realizar el etiquetado de las obs nuevas. 
    En la iteracion 0 yy=mapa, pero en las iteraciones siguientes yy es el mapa final estimado en la iteración ICM
    anterior.
    
    #actualizo (o creo) ubicación de arboles observados

    Parámetros:
    -----------

    Entradas:
     - [2 x Lact] Mapa: Es una matriz con las posiciones 2D de 'Lact' árboles.
     - yy: Es el mismo mapa anterior solo que contiene los árboles observados.
     - [(x,y) x Nobs] obs: Lista de observaciones en coordenadas cartecianas.
       Nobs son la cantidad de observaciones filtradas (sin outliers). 
     - [int] Lact: cantidad de árboles hasta el momento (Landmarks
       activos/actuales)
     - config: parámetros de configuración dado por el método ConfigICM
    Salidas:
     - [2 x Lact] Mapa: Es una matriz con las posiciones 2D de 'Lact' árboles.
     - int [1 x Lact] cant_obs_i: Conteo de la cantidad de veces que se observo
       un árbol.
     - int c: Vector de etiquetas de cada landmark. Dice a que landmark
       corresponde cada medición.
     - [int] Lact: cantidad de árboles actualizado.
    """

    if Lact==0:#este bucle es solamente para t=0 de la iteración ICM 0
        c=fcluster(linkage(pdist(obs)),config.dist_thr)-1  #calculo clusters iniciales
        Lact=np.max(c)+1  #cantidad de arboles iniciales
        for i in range(Lact):
            mapa[:,i]=np.mean(obs[c==i,:],axis=0).T #calculo el centro de cada cluster
            cant_obs_i[i]=len(c[c==i])

    else:
        #me fijo si las observaciones corresponden a un arbol existente
        distancias=cdist(yy[:,:Lact].T,obs)#matriz de distancias entre yy y las obs nuevas
        min_dist=np.amin(distancias,axis=0)#distancia minima desde cada obs a un arbol de yy
        c=np.argmin(distancias,axis=0)#etiqueta del arbol de yy que minimiza la distancia a cada obs nueva
        c[min_dist>config.dist_thr]=-1#si esta lejos de los arboles de yy le asigno la etiqueta -1 momentaneamente
        #armo cluster con observaciones de arboles nuevos
        ztt=obs[min_dist>config.dist_thr,:]#extraigo las obs nuevas que estan lejos de los árboles de yy
        if ztt.shape[0]>1:#si hay mas de una observacion de un arbol no mapeado aun
            cc=Lact+fcluster(linkage(pdist(ztt[:,2:4])),config.dist_thr)-1 #calculo clusters y le coloco una etiqueta nueva (a partir de Lact=max etiqueta+1)
            c[c==-1]=cc#a todos los árboles con etiqueta -1 le asigno su nueva etiqueta

        elif ztt.shape[0]==1:#si hay sólo una observacion de un arbol no mapeado aun
            c[c==-1]=Lact

        Lact=np.amax(np.append(c+1,Lact))#actualizo la cantidad de arboles mapeados hasta el momento
        #actualizo (o creo) ubicación de arboles observados
        for i in range(Lact):
            if len(c[c==i])>0:
                mapa[:,i]=np.sum(zt[c==i,2:4],axis=0)/(cant_obs_i[i]+len(c[c==i]))\
                        +mapa[:,i]*cant_obs_i[i]/(cant_obs_i[i]\
                        +len(c[c==i]))

                cant_obs_i[i]=cant_obs_i[i]+len(c[c==i])

    return mapa,cant_obs_i,c,Lact
