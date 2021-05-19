# juge-tron
## Introducción
Este es el repositorio de un proyecto que trata de un robot en forma de bola que tiene como función jugar con las mascotas huyendo de ellas. Para realizar este propósito, el robot contará con una cámara con la cual, a través de visión por computador, será capaz de visualizar al animal y su entorno para así poder actuar con cierta personalidad. 
El robot se moverá debido a la fricción ejercida por unas ruedas internas contra la carcasa de la bola, y los elementos del interior se mantendrán estables debido a un peso que ejercerá de centro de masas.

## Requisitos
Para ejecutar la simulación hacen falta las siguientes dependencias de python:
```
pip install numpy, opencv-python, tensorflow, 
```
También se requiere tener instalado el simulador Webots.

## Cómo ejecutar la simulación
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
