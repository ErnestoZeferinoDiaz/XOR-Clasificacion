import matplotlib.pyplot as plt
import numpy as np
import red

#entradas y salidas de nuestros datos
X=np.matrix([[0,0],[0,1],[1,0],[1,1]])
Y=np.matrix([[0],[1],[1],[0]])

rows = X.shape[0]

#el numero de capas de nuestra red neuronal y la funcion de activacion que usaran
capas = [X.shape[1],15,Y.shape[1]]
fun = [red.sigmoid,red.sigmoid]

#rango que usaran los pesos sinapticos para generar sus valores aleatorios
minimo = -1
maximo = 1

#paso de aprendizaje
alpha = 0.1

#la red neuronal iterara hasta que el error sea menor a
e=10**(-3)

#retorna una lista con 3 items
# 1) lista de valores del error a lo largo del entrenamiento
# 2) lista con las matrices de pesos sinapticos (las Ws)
# 3) lista con las matrices de los umbrales (las Bs)
resp=red.redNeuronal(minimo,maximo,capas,fun,alpha,X,Y,e)

#generamos valores para generar la frontera de decision
u = np.linspace(-2, 2, 200)
v = np.linspace(-2, 2, 200)
z = np.zeros((200,200))

#inicializamos unas listas de tamaño del numero de capas de la red
a=[]
As=[]
Zs=[]
for i in range(len(capas)):
    As.append(0)
    if(i<len(capas)-1):
        Zs.append(0)

#empezamos a probar el modelo ya entrenado con los nuevos datos
for i,v1 in enumerate(u):
    for k,v2 in enumerate(v):
        As[0] = [[v1],[v2]]
        j=0
        while(j<len(capas)-1):
            Zs[j] = np.dot(resp[1][j],As[j]) + resp[2][j]
            As[j+1] = fun[j](Zs[j])[0]
            j = j+1
        z[i,k]=As[j][0,0]

#Graficamos los resultados
f,grf = plt.subplots(1, 2)

#graficamos el progreso del error, veremos como fue disminuyendo
grf[0].plot(range(len(resp[0])),resp[0],color="blue",linewidth=1)
grf[0].set_title('Progeso del error')

#graficamos la compuerta XOR
grf[1].plot(X[np.where(Y==0)[0],0],X[np.where(Y==0)[0],1],"ro")
grf[1].plot(X[np.where(Y==1)[0],0],X[np.where(Y==1)[0],1],"bo")

#graficamos la frontera de decision
#grf[1].contour(u,v,z,50)
grf[1].pcolormesh(u,v,z)

#rango de vision de la grafica y activacion de la grid
grf[1].axis([-0.5, 1.5, -0.5, 1.5])
grf[1].grid(True)
grf[1].set_title('Compuerta XOR clasificada')
#mostramos la grafica
plt.show()