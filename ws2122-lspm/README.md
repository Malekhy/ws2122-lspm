# Event Log Sampling for Predictive Monitoring
A web app which takes(csv/xes) files as an input and produce sampled (xes) files as an output according to many different sampling methods.


#

## Prerequisite
* Make sure your system have Docker installed 
* Otherwise follow the instruction of setup and installation of Docker from https://docs.docker.com/docker-for-windows/install/

## A - Run Using Virtual Environment:
* 1- Clone the project from https://github.com/Malekhy/ws2122-lspm
* 2- Open the command terminal 'cmd'
* 3- Move into the directory folder of the project: `(project-path)\ws2122-lspm`
* 4- Move into project environment: 
   - **A - For Windows users:** `.\Scripts\activate`
   - **B - For Mac users:** `source Scripts\activate`
* 5- Install requirements: `pip install -r requirements.txt`
* 6- Run the web app: `python manage.py runserver`
* 7- Open the browser and hit the URL: `http://localhost:8000/`

## B - Run Using Docker Image:
* 1- Clone the project from https://github.com/Malekhy/ws2122-lspm
* 2- Move into the directory folder of the project: `(project-path)\ws2122-lspm`
* 3- Open the command terminal 'cmd'
* 4- Build the docker image using this command: `docker build --tag python-lspm .` 
* 5- Run the docker container using this command: `docker run --publish 8000:8000 python-lspm`
* 6- Open the browser and enter this URL: `http://localhost:8000/`

----------------------   Happy Predictive Monitoring   -----------------------
