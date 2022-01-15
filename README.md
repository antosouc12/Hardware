# Hardware
Module Hardware CUDA

Lors des sceances de HSP, nous allons tenter d'implementer une reseau de neurones a la main sur GPU. Nous allons essayer d'implement un LeNet-5.

L'agolorithme LeNet-5 est un reseau de neurones qui permet de classifier des images de chiffres ecrits a la main. 
Ce reseau est compose de deux couches de convolution, deux couches de subsampling et deux couches fully connected.

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



