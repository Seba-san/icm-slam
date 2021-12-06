# icm-slam
a SLAM solved by ICM methodology
ICM SLAM
===================

Instalación ICM_SLAM + ROS
--------------------------
A continuación se desarrollan los pasos que hay que seguir para poder ejecutar el algoritmo ICM_SLAM en una red ROS. El objetivo de este instructivo es guiar a un usuario *no familiarizado* con las distintas tecnologías aquí utilizadas. Cualquier complicación contactarse a la siguiente direccion: **ssansoni at inaut.unsj.edu.ar**

 - Inicialmente es necesario instalar [Docker](https://docs.docker.com/get-docker/)  
 - Descargar la última versión de [ROS_bridge_docker](https://github.com/Seba-san/icm-slam/files/7661068/docker_v2_1.zip)
 - Descomprima el archivo .zip descargado. Abra una terminal del sistema operativo y dirijase a la carpeta en donde descomprimió el archivo.

 - Luego ingrese el siguiente comando:
~~~bash
    docker build -t ros-bridge .
~~~

Se van a descargar unos paquetes, esta operación puede tardar mucho tiempo. Depende de su velocidad de conexión.

 - Una vez que finalice el anterior paso, ejecutar:
 ~~~bash
    docker-compose up -d 
 ~~~
   
Este comando va a crear una red ROS funcionando en background. Es necesario que quede corriendo mientras se trabaja con ROS ya que además de generar un puente entre la virtualización (mediante Docker) y la maquina host, corre tambien *roscore*. Esta última aplicación es necesaria para que funcione ROS.
Para desactivarlo, ingrese el siguiente comando:

~~~bash
    docker-compose down 
~~~

En esta misma carpeta hay otra carpeta que se llama "bags". Esta es una carpeta compartida con la virtualizacion, los datos almacenados en la carpeta "bags" apareceran en la carpeta "/home" de la virtualizacion. Si desea compartir otra carperta debe modificar el archivo docker-compose.yml y en la linea donde dice "source: ./bags" rempleazar ./bags por la ruta de la carpeta que desea compartir. 

 - Para reproducir un archivo *.bag*, en otra terminal posicionarse en la carpeta donde descomprimió el archivo *.zip de docker* y  ejecutar el siguiente comando:
~~~bash
    docker exec -it ${NOMBRE DE LA CARPETA}_ros-bridge_1 bash
~~~

donde ${NOMBRE DE LA CARPETA} es el nombre de la carpeta en donde se encuentra almacenado el archivo *docker-compose.yaml*, por ejemplo:
~~~bash
    docker exec -it docker_ros-bridge_1 bash
~~~

---

Otra forma de ejecutar este comando evitando este conflicto es:

~~~bash     
    docker-compose exec ros-bridge bash 
~~~
     
---
Luego, una vez dentro de la virtualización ingresar los siguientes comandos en orden:

~~~bash
    cd /home
    source iniciar.sh 
~~~


Ahora ya se esta en condiciones de reproducir cualquier archivo .bag, por ejemplo lo puede hacer de la siguiente manera: 
~~~bash
    rosbag play data_IJAC2018.bag
~~~

Esto reproducirá la grabación almacenada en *data.bag* . Si preciona la tecla *space* la repdocucción se pausará. Si preciona la combinación de teclas *ctrl+c* se corta la reproducción.

Una vez de realizado el anterior procedimiento, se esta en condiciones de probar el algoritmo *icm_slam*. Para esto descargar la última versión disponible en [icm_slam](https://github.com/Seba-san/icm-slam/files/7661067/ICM_SLAM_ROS_v2_1.zip) e instalar [Python 3](https://www.python.org/downloads/) si es que no lo tiene aún. Luego siga las siguientes instrucciones:

 - Descomprimir el archivo.
 - En la carpeta que descomprimió el archivo ejecutar los siguientes comandos para preparar el entorno de ejecución:

~~~bash
     python3 -m venv env
     source env/bin/activate
     python -m pip install -U pip
     python3 -m pip install -r requisitos.txt 
~~~

---

En sistemas con Windows, las anteriores lineas son:

~~~bash      
     py -m venv env
     py -m venv env/bin/activate
     py -m pip install -U pip
     py -m pip install -r requisitos.txt
~~~

---
Ya se esta en condiciones de ejecutar el algoritmo. Para esto, debe poner en funcionamiento la virtualización (mediante el comando  `docker-compose up -d`), luego en la carpeta donde descomprimió ejecutar:
   
~~~bash
     python example.py
~~~

Si todo funciona correctamente debería salir el mensaje: "Conectado a la red ROS".

- Cuando se comience a reproducir una base de datos, el código de python comenzará a recibir la información y creará una imagen con la trayectoria y el mapa estimado. 


Flujo de trabajo
-----------------

Una vez que este todo instalado y se compruebe el funcionamiento, el flujo de trabajo sería el siguiente:

- Posicionarse en la carpeta en donde se encuentra el archivo *docker-compose.yaml* y ejecutar:
~~~bash
    docker-compose up -d 
~~~

- Posicionarse en la carpeta donde esta el proyecto *icm slam* (en su interior tiene una carpeta con el nombre *env*) y luego ejecutar los siguientes comandos en orden:

~~~bash
    source env/bin/activate
    python example.py
~~~
Si todo esta funcionando bien, tendría que salir el mensaje: "Conectado a la red ROS". 

- Ahora se está en condiciones de comenzar a reproducir cualquier archivo *.bag*, para esto en una nueva terminal ejecutar los siguientes comandos:
  
~~~bash 
    docker-compose exec ros-bridge bash
    cd /home
    source iniciar.sh
    rosbag play data_IJAC2018.bag
~~~


Procedimiento de ejecución sin Docker
--------------------------------------

Este package no es propiamente un package de ROS, ya que no requiere tener inicializado el entorno. Para usar los scripts con la red ROS se requiere estar corriendo en *ros-master* el modulo [rosbridge](http://wiki.ros.org/rosbridge_suite). Una vez instalado este modulo ejecutar:
~~~bash
    roslaunch rosbridge_server rosbridge_websocket.launch
~~~

A modo de resumen, rosbridge es una inteerfaz de ROS que permite conectar a la red programas que no contienen modulos de ROS. En este caso se utiliza la libreria [roslibpy](https://roslibpy.readthedocs.io/en/latest/) ya que permite conectar códigos escritos en python 3.xx a ROS (ROS funciona con python 2.7). 

Una vez inicializado *rosbridge* el módulo *example.py* busca conectarse a la red ROS y accede a los topics de los sensores. Por defecto busca el topic del laser y de la odometría, los cuales deben especificarse en el archivo de configuración *config_ros.yaml* en *topic_laser* y *topic_odometry* respectivamente.

El software no diferencia si los datos vienen de una simulación que esta corriendo, si son datos de un robot que está compartiendo los datos de sus sensores via streaming, o son datos grabados en un archivo *.bag*. Por este motivo en esta ocasión se utiliza la última opción. 

Ya sea con *ros_bridge* implementado en Docker, o directamente desde la máquina host, para reproducir un archivo *.bag* hacer:

<!--Aún no está implementada la posibilidad de de ejecutar un *.bag* sin tener funcionando la red ROS. Con lo cual, se requiere tener instalado ROS para ejecutar el siguiente comando:-->

~~~bash
    rosbag play slam_benchmark/bags/laser_data.bag
~~~

Asumiendo que el *PWD* está en la carpeta *src* del *workspace*. Luego lanzar el script haciendo:

~~~bash
    source ICM_SLAM/scripts/env/bin/activate
    python ICM_SLAM/scripts/example.py
~~~

Funcionamiento script example.py
----------------------------------

Este script se subscribe a los topics antes dichos y genera un servicio que habilita las iteraciones ICM. (Esto se podrá implementar en un futuro con  [ActionLib](http://wiki.ros.org/actionlib) más info en [ActionLib Detailed Description](http://wiki.ros.org/actionlib/DetailedDescription>)).

Básicamente el comportamiento del servicio es que mediante el comando "start", comienza a iterar la cantidad de veces especificadas en el *.yaml* y luego devuelve el mapa y la trayectoria del vehículo. Para dar el comando mediante la red ROS, ejecutar la siguiente línea:
   ~~~bash
   rosservice call /icm_slam/iterative_flag true
   ~~~

Esquema de las funciones implementadas
--------------------------------------

A modo de resumen a continuación se muestra un esquema del funcionamiento del módulo implementado

![Esquema general][./esquema_general.png]
