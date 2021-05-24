# Juge-tron

# Tabla de contenidos
<li>
<a href="#whatis"> ¿Qué es este proyecto? </a>
</li>
<li>
<a href="#req"> Requerimientos </a>
</li>
<li>
<a href="#docu"> Documentación </a>
</li>
<li>
<a href="#howtouse"> Cómo utilizar </a>
</li>
<li>
<a href="#design" id="#design"> Diseño del robot </a>
</li>

##  ¿Qué es este proyecto?
Este es el repositorio de un proyecto que trata de un robot en forma de bola que tiene como función jugar con las mascotas huyendo de ellas. Para realizar este propósito, el robot contará con una cámara con la cual, a través de visión por computador, será capaz de visualizar al animal y su entorno para así poder actuar con cierta personalidad. 
El robot se moverá debido a la fricción ejercida por unas ruedas internas contra la carcasa de la bola, y los elementos del interior se mantendrán estables debido a un peso que ejercerá de centro de masas.

## Requerimientos
Para ejecutar la simulación hacen falta las siguientes dependencias de python:
```
pip install numpy, opencv-python, tensorflow, 
```
También se requiere tener instalado el simulador Webots.
## Documentación
Los siguientes proyectos han servido como referencia e inspiración:
https://www.hindawi.com/journals/complexity/2019/7543969/
https://www.sciencedirect.com/science/article/pii/S0307904X14006015
https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb
https://nanonets.com/blog/optical-flow/

## Cómo utilizar
1. Clonar el repositorio
2. Instalar las siguientes dependencias
```
pip install -U --pre tensorflow=="2.*"
pip install tf_slim
```
3. Dirigirse a la carpeta "/Reconocimiento de imágenes" y ejecutar los siguientes comandos:
``` powershell
cd research/
protoc object_detection/protos/*.proto --python_out=.
pip install .
```
4. Abrir haciendo doble click uno de los escenarios presentes en /RLP_Sim/worlds

## Diseño del robot 
<a href="#design" name="#design"></a>

En esta carpeta se incluye el diseño que tendrá el robot.
### Componentes
Los componentes Hardware utilizados serán los siguientes:

![components](Imágenes/precios.JPG)

El tamaño de cada componente se incluye en la siguiente tabla:

![tamany](Imágenes/Tamaños.JPG)

### Esquema 3D
El esquema básico dentro de la bola se compone de una base cilíndrica, sobre la que se colocarán los componentes hardware necesarios, tres ruedas que harán girar la carcasa de la bola mediante la fricción que ejerzan sobre ella, y tres varitas que unan las ruedas a la base. Este esquema se muestra en la siguiente imagen:

![base](Imágenes/esquema_base.JPG)

las piezas aquí mostradas se obtendrán mediante impresión 3D. Las medidas, colocación y ángulos relevantes de estas se indican en el esquema. El diseño 3D de las varitas incluye un pequeño soporte para poder pegarlas a la base con pegamento, un espacio para poder colocar las ruedas, y un agujero con una pieza que permite fijar las ruedas a la varita.

Debajo de la base se pegará un cilindro. Contendrá un peso que ejercerá como centro de masas para estabilizar el robot internamente. Este cilindro también se obtendrá con impresión 3D.

![base](Imágenes/esquema_carcassa.JPG)

Finalmente, encima de la base se colocarán los diferentes componentes Hardware de manera estratégica.

![base](Imágenes/esquema_final.JPG)

Las ruedas se conectarán a los motores mediante un sistema de poleas. El diseño de las piezas 3D se incluye en la carpeta piezas, y en el archivo Esquema3d.dwg se ha incluido un modelo 3D del esquema de colocación de las piezas que aquí se ha explicado.

### Esquema Hardware
El esquema de conexión entre los diferentes componentes hardware es el siguiente

![base](Imágenes/esquema.png)


