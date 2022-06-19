# DrowsinessPi
A continuación se describirá el proceso de instalación de librerías y los
pasos requeridos para poder ejecutar el script de detección de somnolencia
en la Raspberry Pi 4. Se asume que ya se encuentra instalado y configurado
el sistema operativo Raspberry Pi OS en una tarjeta MicroSD, el cual puede
ser descargado desde el [sitio oficial](https://www.raspberrypi.com/software/).


**1. Clonado del repositorio**

En primera instancia, se debe clonar el repositorio DrowsinessPi desde
GitHub en la Raspberry Pi, en donde se encuentran tanto el script de detección mencionado anteriormente, como aquellos utilizados para la generación
de los dos datasets y el entrenamiento del modelo. Para hacerlo, se debe
ejecutar el siguiente comando en la terminal:
```
git clone https://github.com/DevTeamCR/DrowsinessPi.git
```

**2. Instalación de librerías**

Una vez clonado el repositorio, deben instalarse las librerías necesarias ejecutando cada comando detallado a continuación:

```
cd DrowsinessPi/Detección 
pip install -U -r requirements.txt 
sudo apt-get install libcblas-dev 
sudo apt-get install libhdf5-dev 
sudo apt-get install libhdf5-serial-dev 
sudo apt-get install libatlas-base-dev 
sudo apt-get install libjasper-dev 
```
**3. Ejecución**

Una vez instaladas las librerías anteriores, dentro del directorio "Detección"
se debe ejecutar el comando ```python deteccion.py``` para dar inicio al proceso
de detección, el cual abrirá una ventana mostrando la captura realizada por
la cámara. Para finalizar la ejecución del script, se debe presionar ```Ctrl+C```.

***

:computer: *Desarrollado por ***Macarena Quiroga*** y ***Emilio Melo****
