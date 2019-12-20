# Fall-2019-final-project

# Two ML/ CV models are implemented and flask applications for both are deployed
#### Weight visualization technique is implemented and itegrated in the main visualisation application whereas the Deep Dream which is a tool for visualisation is simply deployed on IBM Kubernetes


# Build the docker image inside vagrant and run like:
* docker build -t deep_dream .

# If getting memory error then run
* docker system prune -a 

# Run the docker
* docker run -p 8001:8001 -it deep_dream

# To see the output, open a terminal and give the image using curl
* Run this command from the folder where test image is saved
* curl -X POST -F image=@spider.png "http://localhost:8001/"
* Output is shown on terminal

# Keep the port name consistent in Dockerfile and flask file

#  Change the docker image name tag in Vagrant file, halt and reload the vagrant
* vagrant halt -f
* vagrant reload --provisions


#  Create account on docker hub, push the docker image from vagrant on Docker hub and make your repository public
* docker login --username=divajuneja 
* docker images
* docker tag 9fb09e33782b divajuneja/deepdream:v1
* docker push divajuneja/deepdream
* Link: https://hub.docker.com/repository/docker/divajuneja/deepdream


#  Go to kubernetes dashboard using IBM portal and create the yaml file, give the name of docker file 

#  Wait till the pods are created, check the status if running
* kubectl get pods


#  Expose it to the port
* kubectl expose deployment deepdream-deployment --port 8001 --target-port 8001 --type=NodePort


# Check the Public IP from IBM cloud Kubernetes cluster created

# Check the Public IP of the container from the output of
* kubectl describe svc deepdream-deployment
* NodePort:                 <unset>  32361/TCP

# Pass the image using curl from the folder where test image is saved using the ip got by above command
* curl -X POST -F image=@IMG_0098.jpg "http://184.173.1.97:32361/"
* Output is shown on the terminal.
* Image will be saved in the same folder from which curl command was run and where the input image exists.

## Application of weight visualisation is directly deploed on IBM Kubernetes using Git

## All of the code of weight visualisation is integrated into one main application