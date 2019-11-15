import numpy as np
import matplotlib.pyplot as plt

#funciones de activacion, retornan una lista con dos items
# 1) la funcion de activacion
# 2) la derivada simple de la funcion de activacion 

# 1) y = x     2) y' = 1
def lineal(x):
    return [x,np.ones(x.shape)]

# 1) y = x     2) y' = 1   si x>0
def relu(x):
    y = np.abs(x)
    z = (y + x)/2
    return [z,np.where(z>0,1,0)]

# 1) y = 1/(1+e^(-x))     2) y'= y*(1-y)
def sigmoid(x):
    return [1.0/(1.0+np.exp(-x)),np.multiply(x,(1-x))]

# 1) y = (e^(x)-e^(-x))/(e^(x)+e^(-x))     2) y'= 1-y^2
def tanh(x):
    ep=np.exp(x)
    en=np.exp(-x)
    return [(ep-en)/(ep+en),1-np.power(x,2)]

# 1) y = (e^(x)-e^(-x))/(e^(x)+e^(-x))     2) y'= 1-(1/e^y)
def rectif(x):
    #return [np.log(1+np.exp(x)),1.0/(1.0+np.exp(-x))]
    return [np.log(1+np.exp(x)),1-(1.0/(np.exp(x)))]

def redNeuronal(minimo,maximo,capas,fun,alpha,X,Y,error):
    emedio=[] #lista para guardar los valores del error medio
    Ws=[] #lista para guardar las matrices de pesos sinapticos
    Bs=[] #lista para guardar las matrices de umbrales
    As=[] #lista para guardar las matrices de las salidas de las capas
    Zs=[] #lista para guardar las matrices de los valores netos
    Ss=[] #lista para guardar las matrices de sensibilidades
    
    rows = X.shape[0] #con cuantos registros estamos trabajando
    
    #inicializamos las listas
    for i in range(len(capas)):
        As.append(0)
        if(i<len(capas)-1):
            Ws.append(minimo + np.random.rand(capas[i+1],capas[i]) * (maximo - minimo))
            Bs.append(minimo + np.random.rand(capas[i+1],1) * (maximo - minimo))

            #lineas comentadas que sirven para leer pesos y umbrales previamente guardados
            #Ws.append(np.matrix(np.load("modelo/W"+str(i)+".npy")))
            #Bs.append(np.matrix(np.load("modelo/B"+str(i)+".npy")))
            Zs.append(0)
            Ss.append(0)
    
    eI=1000 # decimos que inicialmente nuestra red tiene un error de 1000
    epocas=0
    while(eI>error): #mientras la variable eI sea mayor alo que le indicamos en el parametro error entonces
        suma=0
        for i in range(rows):#recorremos todos los registros, todas nuestras entradas
            As[0] = X[i,:].transpose() #primera entrada

            #propagacion hacia adelate
            j=0
            while(j<len(capas)-1):
                Zs[j] = np.dot(Ws[j],As[j]) + Bs[j]
                As[j+1] = fun[j](Zs[j])[0]
                j = j+1

            # calculamos el error
            e = Y[i,:].transpose()-As[j]
            
            #propagacion hacia atras, calculo de las sensibilidades
            j=j-1
            while(j>=0):
                if(j==len(capas)-2):
                    Ss[j] = -2*(fun[j](As[j+1])[1])*e
                else:
                    tmp1 = fun[j](As[j+1])[1]
                    Ss[j] = np.diagflat(tmp1)*Ws[j+1].transpose()*Ss[j+1]
                j = j-1
            #actualizacion de pesos sinapticos
            j=0
            while(j<len(capas)-1):                
                Ws[j] = Ws[j] - alpha*Ss[j]*As[j].transpose()
                Bs[j] = Bs[j] - alpha*Ss[j]        
                j = j+1      
            
            #elevamos al cuadrado la matriz de error
            suma = e.transpose()*e + suma

        #sacamos el promedio del error de entre todos los registros
        emedio.append((suma/rows)[0,0]) 
        eI=emedio[epocas] #actualizamos el error
        epocas=epocas+1 
        print(epocas,": ",eI)
    # una vez terminado de iterar guardamos los valores de las matrices de pesos sinapticos y umbrales
    j=0
    while(j<len(capas)-1):
        # guardamos en formato npy, extension por parte de la libreria numpy
        # importante guardarlo asi porque con csv algunas matrices se trasponen y causa conflicto
        # ya que con la extension npy no podemos visualizar los valores en algun editor
        # procedemos a guardarlo de igual manera en un archivo .csv
        np.save("modelo/W"+str(j),np.int_(np.around(Ws[j],0)))
        np.save("modelo/B"+str(j),np.int_(np.around(Bs[j],0)))
        np.savetxt("modelo/W"+str(j)+".csv",np.int_(np.around(Ws[j],0)),delimiter=",")
        np.savetxt("modelo/B"+str(j)+".csv",np.int_(np.around(Bs[j],0)),delimiter=",")
        j = j+1
    return [emedio,Ws,Bs]