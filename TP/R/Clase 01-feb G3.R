
#--------------------------------
# Curso: Teoría de Portafolio
# Grupo 3
# Codigo 01 y 03 de febrero
#-------------------------------

# Instalar librerias

install.packages('zoo')
install.packages('xts')
install.packages('quantmod')

# Cargar librerias

library(zoo)
library(xts)
library(quantmod)

# Importar serie de precios

mdate="2020-01-01"
getSymbols('BAC',from=mdate)

PBAC <- cbind(BAC[,6],BAC[,4]) # Trabjamos con precios ajustados

quartz() # windows()
plot(PBAC)

chart_Series(PBAC) # Gráfico de vela

# Calculo de los retornos:

# rt = Ln(Pt/Pt-1): Ln(Pt) - Ln(Pt-1) => Primera diferencia en logartimos

rBAC <- diff(log(PBAC))[-1,] # [-1,]: elimina la fila vacia
rBAC2 <- ROC(PBAC)[-1,] # ROC permite también el calculo de retornos discretos

# rBAC y rBAC2 deben ser iguales

# quartz() # 
windows()
plot(rBAC2[,1])

# Grafico de distribucion - histograma de frecuencias

# quartz() # windows()
windows()
hist(rBAC2[,1],breaks = 30,probability = T)
lines(density(rBAC2[,1]),col="red")

# Distribución normal

# Densidad: dnorm - PDF: Función de Densidad de Probabilidad

x <- seq(-5,5, by=0.1)
y <- dnorm(x,mean=0,sd=0.5)

# quartz() # windows()
windows()
plot(x,y, type = "l")

# CDF: Función de Densidad Acumulada

z <- pnorm(x, mean = 0,sd=0.5)

# quartz() # windows()
windows()
plot(x,z, type="l")

# Función inversa

p <- seq(0,1,by=0.01)
i <- qnorm(p,mean=0,sd=0.5)
# quartz() # windows()
windows()
plot(i,p,type="l")

# Probabilidades mportantes: 1% y 5%

qnorm(0.01,mean=0,sd=1) # Normal estandar
qnorm(0.05,mean=0,sd=1) # Normal estandar

# Numeros aleatorios
n <- 1000  # Tamaño de la muestra

r <- rnorm(n)
# quartz() # windows()
windows()
hist(r,breaks=30,probability = T)
lines(density(r),col="blue")

# Agregar la función teorica (normal)
x <- rnorm(100000)
# quartz() # windows()
windows()
hist(x,breaks=20,probability = T)
lines(density(x),col="blue")
curve(dnorm(x),add=T,col="red")












  
  
  
  
  
  
  