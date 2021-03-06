
#--------------------------------
# Curso: Teoría de Portafolio
# Grupo 4
# Sesión 16 de febrero
#-------------------------------

# Markowitz (1952) - Modelo Media Varianza (MV)
# Algoritmo de optimizacion - Formulación general del problema de optimización

# Carga de librerias
library(xts)
library(zoo)
library(quantmod)

rm(list=ls()) #Limpia el Environment de R

# Carga de datos
activos <- c("TSLA","BAC","MA","KO","F") # Se pueden ingresar n activos
fecha1 <- "2011-02-01"
fecha2 <- "2022-02-01"
periodicidad <- "monthly" # También puede ser: daily, weekly

#Creación de precios a partir de los activos definidos en "activos"
precios <- xts()
for(i in 1:length(activos)){
  aux <- Ad(getSymbols(activos[i],from=fecha1,to=fecha2,  #La función Ad solo toma Precio ajustado
                       periodicity=periodicidad,auto.assign=FALSE))
  aux <- na.approx(aux,na.rm=FALSE) # Interpolacion de datos con NA
  precios <- cbind(precios,aux)
}

precios

colnames(precios) <- activos # Pega encabezado de nombres en "precios" - fila título
tclass(precios) <- "Date"  # Formato de fecha en "precios" - columna título

plot(precios[,1:3], type = "s", main = "Precios", lwd = 4)

# Modelo MV
retornos <- diff(log(precios))[-1] #[-1]: na.omit 
cov <- cov(retornos) # Matriz Var-Cov; para anualizar, operar cov*12
mu <- colMeans(retornos) # Retorno esperado para cada activo
var <- diag(cov)      # Varianza
sigma <- sqrt(var)    # Desviacion estandar 

#Construcción de los portafolios optimos

n <- length(mu)
ones <- rep(1,n)
x <- t(mu)%*%solve(cov)%*%mu   # solve: Genera la matriz inversa
y <- t(mu)%*%solve(cov)%*%ones
z <- t(ones)%*%solve(cov)%*%ones
d <- x*z - y^2

solve(cov,ones)

g <- (solve(cov,ones)%*%x-solve(cov,mu)%*%y)%*%solve(d)
h <- (solve(cov,mu)%*%z-solve(cov,ones)%*%y)%*%solve(d)

nport <- 1000
Rp <- seq(min(mu),max(mu),length=nport) # Retorno de los portafolios
wpo <- matrix(0,ncol=n,nrow=nport) # Matriz de pesos - Variables endogenas
sigmapo <- matrix(0,nrow=nport) # Riesgo de los portafolios (varianza)
rpo <- matrix(0,nrow=nport) # Retorno de los portafolios - Verificación de RP


for(i in 1:nport){
  wi <- g + h * Rp[i]
  sigmapo[i] <- sqrt(t(wi)%*%cov%*%wi)
  rpo[i] <- t(wi)%*%mu
  wpo[i,] <- t(wi)
}


# PMV
wpmv <- solve(cov,ones)%*%(1/z) #
rpmv <- t(wpmv)%*%mu
sigmapmv <- sqrt(t(wpmv)%*%cov%*%wpmv)

#quartz()
plot(sigma,mu,main="Plano Riesgo-Retorno",xlim=c(0,max(sigma*1.1)),ylim=c(0,max(mu*1.3)),col="red")
lines(sigmapo,rpo,col="blue", lwd = 4)
points(sigmapmv,rpmv,lwd = 4,col="red")
text(sigmapmv,rpmv,labels="PMV",pos=2)
text(sigma,mu,labels=activos,pos=4,cex=0.8)

rownames(wpmv) <- activos

#quartz()
barplot(t(wpmv),main="Pesos PMV",col="orange")
