import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = True

data= np.genfromtxt("datos-carro-1d.csv",dtype=float,skip_header=5,missing_values=np.nan,delimiter=",")
def test(t,a,b,c):
    return a+b*t+(c/2)*t**2
mov=data[0:,1]>=0.001
fit=data[mov]
print("El tiempo t0 es:\t",f"{np.amin(fit[0:,0])} segundos")
t=fit[:,0]
x=fit[:,1]
plt.figure(1)
plt.scatter(data[:,0],data[:,1],c="k",s=10,label="Data")
plt.scatter(min(t),min(x),c="r",s=10,label=r"$t_0$")
plt.xlim((0,max(data[:,0]+0.03)))
plt.grid(True)
plt.xlabel(r"t[s]")
plt.ylabel(r"x[m]")
plt.title(r"$t[s] vs.\; x[m]$")
plt.legend(loc="upper left")
plt.show()

#Se selecciono el t0 como 1.066s, o como el data[31,0]. Consecuente, el ajuste lineal.
param, paramcov = curve_fit(test, t, x)
print("Parametros del ajuste:\n",f"ax={param[2]:.6f}\u00B1{paramcov[2,2]:.6f}\n"
      ,f"vx={param[1]:.5f}\u00B1{paramcov[1,1]:.5f}\n",f"x0={param[0]:.6f}\u00B1{paramcov[0,0]:.6f}")
yfit=(param[0]+param[1]*t+(param[2]/2)*t**2)
#Calcular el promedio de ax de la tabla de datos, y comparar porcentualmente.
ax=np.mean(fit[:50,5]) #Aqui se selecciona hasta la fila 50, para evitar los nan, y se calcula el promedio de los valores.
print(f"Aceleración obtenida de la tabla:\t{ax:.6f}")
er=((ax-param[2])/param[2])*100
print(f"El error porcentual es de:\t{er:.4f}")

#Graficar
plt.figure(2)
plt.scatter(t,x,c="k",s=10,label='Data')
plt.plot(t,yfit,c="b",label=r"Ajuste")
plt.xlabel(r"t[s]")
plt.xlim((1.03,max(data[:,0])+0.03))
plt.ylabel(r"x[m]")
plt.legend(loc="upper left")
plt.title(r"Ajuste")
plt.grid(True)
plt.show()