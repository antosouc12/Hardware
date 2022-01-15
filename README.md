# Hardware
Module Hardware CUDA

# TP Hardware for Signal Processing

# Prise en main de CUDA

Au cours des séances de HSP, nous allons tenter d'implémenter une réseau de neurones 'a la main' sur GPU, le LeNet-5.

L'algorithme LeNet-5 est un réseau de neurones qui a été créé dans le but de classer des images de chiffres manuscrits. 
Ce réseau est constitué de trois couches convolutives, de deux couches de subsampling et deux couches fully connected.

Pour ce faire, nous allons utiliser le language CUDA qui permet de coder sur le GPU. Cela va nous permettre de paralléliser les calculs et ainsi de grandement réduire le temps d'exécution des différents programmes.

Voici une video qui illustre bien cette idée:
https://www.youtube.com/watch?v=-P28LKWTzrI&ab_channel=NVIDIA

Le GPU fonctionne de la manière suivante:

![image](https://user-images.githubusercontent.com/56081832/149636392-fc8a8165-ed5f-49ca-9236-af45bd4c419d.png)

Dans le GPU se trouvent des "Blocks" dans lesquelles se trouvent des "Threads". Chaque thread peut effectuer des calculs en parrallèle avec chaque autre Thread de chaque Block.
Cela nous permet, par example, de calculer tous les éléments d'une matrice en même temps.

Le rapport du temps de calcul entre le CPU et le GPU peut etre de l'ordre de 1000. 
On peut utiliser la commande nvprof pour avoir des mesures de temps détaillées lors de l'exécution d'un programme.

Pour pouvoir utiliser ces threads et blocks du GPU, nous devons coder en CUDA. 
Ce language est très proche du C mais avec des libraries en additionnelles  pour coder sur GPU.

Parmi ces nouvelles fonctionnalités, on ajoute de nouveaux prefixes aux fonctions:
__host__ , __global__ et __device__

Ces préfixes permettent au compilateur de savoir dans quelle milieu exécuter la fonction.

Une fonction avec le préfixe "host" indique au compilateur d'exécuter la fonction depuis le CPU dans le CPU.

Une fonction avec le préfixe "global" indique au compilateur d'exécuter la fonction depuis le CPU dans le GPU.

Une fonction avec le préfixe "device" indique au compilatuer d'exécuter la fonction depuis le GPU dans le GPU.

Lors du codage, en plus d'allouer de la memoire dans le CPU, nous devons aussi allouer de la memoire dans le GPU pour stocker les valeurs qui y sont calculées. 
Voici un example de code permet cela:

    float *raw_data;    
    raw_data=(float*)malloc(n*p*sizeof(float));
    float *M_d;     
    cudaMalloc((void **) &M_d,size_f*n*p);

M_d est l'adresse de la mémoire allouée dans le GPU.

Nous devons aussi copier l'information d'un environnement vers une autre. GPU-->CPU ou CPU-->GPU.
Nous avons des fonctions cudaMemCpy qui nous permettent cela:

    cudaMemcpy(raw_data,M_d,n*p*size_f,cudaMemcpyDeviceToHost);
    
Finalement, pour appeler une fonction "global" nous devons utiliser une synthaxe speciale:

     FillMatrix<<<Nblock,Nthread>>>(M_d,n*p);

Nous devons utiliser des "<<<" et ">>>" entre lesquelles nous precisons le nombre de Blocks et de Threads que nous voulons utiliser pour cette fonction. 
Notons que l'on ne peut pas avoir plus de 1024 thread.


# Implementation de LeNet-5

Voici une illustration du reseau LeNet-5:
![image](https://user-images.githubusercontent.com/56081832/149637013-b4aeb829-f86e-49e6-8455-15c23cf95750.png)

Les deux fonction importantes a coder sont: la convolution de deux matrices et la moyen pooling (ou meanpooling) d'une matrice.
Pour coder chaque fonction sur GPU, l'entier idx sera essentiel:

    int idx=blockIdx.x*blockDim.x+threadIdx.x;

Il nous permet d'avoir l'identifiant du thread du block dans lequel nous nous situons. C'est avec cela que l'on peut déterminer le coefficient de la matrice a calculer.

Une fois ces deux fonctions codées, nous pouvons mettre en place notre reseau.

# LeNet-5 sur Python 

A la place d'entrainer notre reseau a la main, nous allons mettre en place un réseau de neurone en python en utilsant tensorflow et keras, entrainer ce modele et exporter les poids.



