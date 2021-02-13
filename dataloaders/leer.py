__author__ = 'alvaro'

import numpy as np
import numpy

#texto= open('eva_soriano.txt')

# no se ha declarado x_vector ni se sabe el nombre del archivo de texto'''
with open("../frontend/iberspeech2020/DATA/xvector/enrollment-test/eva_soriano.txt", 'r', encoding="utf-8") as texto:
    text = texto.read()
    #text.str()
sol = open("../frontend/iberspeech2020/solucionamano", 'w')


numero =1
no_quedan_muestras = True
x_vector = 0
'''
sample = np.array([])
etiqueta = np.array([])
'''
sample = []
etiqueta = []
x = text.find("0")  # se necesita buscar solo una vez
nombre = text[0:x-1]


b = 0
a = 1
encuentraetiqueta = False
indice = 0
numerototalbusqueda=text.count("[")



while indice<len(text):
    if (text[indice]=="["):
        a = indice
        encuentraetiqueta=True
    if (text[indice]=="]"):
        b = indice+1
    if(encuentraetiqueta):
        sample.append(numero)
        numero+=1
        encuentraetiqueta = False
        if (a > b):
            etiqueta.append(text[b:a])
    indice+=1



numerototal= len(sample)

'''
print(numerototal)
print(sample)
print(etiqueta)
'''

DIM = 512	# Numero de dimensiones de cada vector
zero = numpy.zeros(DIM)	# Array de ceros (0, 512)
tamaño= DIM*5


xvector1 = zero
xvector2 = zero
xvector3 = zero
xvector4 = zero
xvector5 = zero

xvector= numpy.zeros(tamaño)

f=0

while f< numerototal:
    if(f-2>=0):
        c1 = text.find(etiqueta[f-2])
        f1 = text.find(etiqueta[f-1])
        xvector1 = text[(c1 + (len(etiqueta[f-2]))+2):f1-2]
        xvector1 = [float(i) for i in xvector1.split(' ')]

    else:
        xvector1= zero


    if (f - 1 >= 0):
        c2 = text.find(etiqueta[f - 1])
        f2 = text.find(etiqueta[f ])
        xvector2 = text[(c2 + (len(etiqueta[f-1]))+2):f2-2]
        xvector2 = [float(i) for i in xvector2.split(' ')]
    else:
        xvector2 = zero


    if(f+2<=numerototal):
        c3 = text.find(etiqueta[f])
        f3 = text.find(etiqueta[f + 1])
        xvector3 = text[(c3 + (len(etiqueta[f]))+2):f3-2]
        xvector3 = [float(i) for i in xvector3.split(' ')]

        '''try:
            xvector3_float=[float(i) for i in xvector3.split('  ')]
        except ValueError as t:
            print ("error",t)
            '''


    else:
        c3 = text.find(etiqueta[f])
        f3 = len(text)
        xvector3 = text[(c3 + (len(etiqueta[f])) + 2):f3 - 3]
        xvector3 = [float(i) for i in xvector3.split(' ')]
    if(f+3<numerototal):
        c4 = text.find(etiqueta[f+1])
        f4 = text.find(etiqueta[f + 2])
        xvector4 = text[(c4 + (len(etiqueta[f+1])+2)):f4-2]
        xvector4 = [float(i) for i in xvector4.split(' ')]
    elif(f+3==numerototal):
        c4 = text.find(etiqueta[f+2])
        f4 = len(text)
        xvector4 = text[(c4 + (len(etiqueta[f+1]))+2):f4-3]
        xvector4 = [float(i) for i in xvector4.split(' ')]
    else:
        xvector4 = zero
    if(f+4<numerototal):
        c5 = text.find(etiqueta[f+2])
        f5 = text.find(etiqueta[f + 3])
        xvector5 = text[(c5 + (len(etiqueta[f+2]))+2):f5-2]
        xvector5 = [float(i) for i in xvector5.split(' ')]
    elif(f+4==numerototal):
        c5 = text.find(etiqueta[f+3])
        f5 = len(text)
        xvector5 = text[(c5 + (len(etiqueta[f+2]))+2):f5-3]
        xvector5 = [float(i) for i in xvector5.split(' ')]


    else:
        xvector5 = zero

    #xvector= numpy.concatenate((xvector1, xvector2, xvector3, xvector4, xvector5), 0)
    #concatenacion= xvector1, xvector2, xvector3, xvector4, xvector5
    concatenacion = numpy.stack((xvector1, xvector2, xvector3,xvector4,xvector5))
    numpy.save('fichero'+ str(f+1), concatenacion, allow_pickle=True)

    '''
    #jj=numpy.stack((xvector1, xvector2))
    # numpy.array([1.2, "abc"], dtype=object)
    # xvector[f].append(xvector[f],xvector1)
    # xvector[f] = numpy.stack((xvector1, xvector2), 0)
    
    xvector[f] = numpy.stack(( xvector1, xvector2, xvector3,xvector4, xvector5,))
    '''

    f += 1


i=0

with open("../../mi procedimiento/sol.txt", 'w', encoding="utf-8") as salida:
    while i<= numerototal:
        print(sample[i], nombre,'fichero' + str(i+1) ,etiqueta[i]  ,"\n")
        salida.write(f'{sample[i]}  {nombre} {etiqueta[i]} '"\n")
        i += 1


        #sol.write(sample[i] , nombre , + etiqueta[i]+ "\n")
        #salida.write(sample[i]  , nombre , etiqueta[i])
        #salida.write("{}\n".format(sample[i]  , nombre , etiqueta[i]))



