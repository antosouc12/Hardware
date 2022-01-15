# Hardware
Module Hardware CUDA

# TP Hardware for Signal Processing

Lors des sceances de HSP, nous allons tenter d'implementer une reseau de neurones a la main sur GPU. Nous allons essayer d'implement un LeNet-5.

L'agolorithme LeNet-5 est un reseau de neurones qui permet de classifier des images de chiffres ecrits a la main. 
Ce reseau est compose de trois couches de convolution, deux couches de subsampling et deux couches fully connected.

Pour ce faire, nous allons utiliser le language CUDA qui permet de coder sur le GPU. Le but de coder sur GPU est la capacite de faire beaucoup de calcul en parrallele.

Voici une video qui illustre bien cette idee:
https://www.youtube.com/watch?v=-P28LKWTzrI&ab_channel=NVIDIA

Le fonctionnement du GPU est le suivant:

![image](https://user-images.githubusercontent.com/56081832/149636392-fc8a8165-ed5f-49ca-9236-af45bd4c419d.png)

Dans le GPU se trouvent des "Blocks" dans lesquelles se trouvent des "Threads". Chaque peut effectuer des calculs en parralleles avec chaques autre Thread de chaque Block.
Cela nous permet de, par example, calculer tous les elements d'une matrice en meme temps.

Le rapport du temps de calcul entre le CPU et le GPU peut etre de l'ordre de 1000. 
On peut utiliser la commande nvprof poru avoir des mesures de temps detaille lors de l'execution d'un programme.

Pour pouvoir utiliser ces threads et blocks du GPU, nous devons coder dans le language CUDA. 
Ce language est quasiment du C mais avec des library en plus pour coder sur le GPU.

Parmi ces nouvelles fonctionnalites, on trouve de nouveaux prefixes aux fonctions:
__host__ , __global__ et __device__

Ces prefixes permettent au compilateur de savoir dans quelle milieu executer la fonction.

Une fonction avec le prefixe "host" indique au compilateur d'executer la fonction depuis le CPU dans le CPU.

Une fonction avec le prefixe "global" indique au compilateur d'executer la fonction depuis le CPU dans le GPU.

Une fonction avec le prefixe "device" indique au compilatuer d'executer la fonction depuis le GPU dans le GPU.

Lors du codage, en plus d'allouer de la memoire dans le CPU, nous devons aussi allouer de la memoire dans le GPU pour stocker les valeurs qui y sont calculees. 
Voici un example de code qui fait cela:

    float *raw_data;    
    raw_data=(float*)malloc(n*p*sizeof(float));
    float *M_d;     
    cudaMalloc((void **) &M_d,size_f*n*p);

M_d est l'adresse de la memoire allouer dans le GPU.

Nous devons aussi copier l'information d'un environnement vers une autre. GPU-->CPU ou CPU-->GPU.
Nous avons des fonctions cudaMemCpy qui nous permettent cela:

    cudaMemcpy(raw_data,M_d,n*p*size_f,cudaMemcpyDeviceToHost);
    
Dernierement, pour appeller une fonction "global" nous devons utiliser une synthaxe specialle:

     FillMatrix<<<Nblock,Nthread>>>(M_d,n*p);

Nous devons utiliser des "<<<" et ">>>" entre lesquelles nous precisons le nombre de Block et de Thread que nous voulons utiliser pour cette fonction. 
On note que l'on peut pas avoir plus de 1024 thread.


Avec ces outils, nous pouvons essayer de d'implementer ce reseau sur GPU.

Voici une illustration du reseau LeNet-5:
![image](https://user-images.githubusercontent.com/56081832/149637013-b4aeb829-f86e-49e6-8455-15c23cf95750.png)

Les deux fonction importantes a coder sont la convolution et la moyen pooling.
Dans les deux cas, ce qu'il va nous est essentiel lors du codage sur GPU est:



