
#--------------------------------
# Curso: Teoría de Portafolio
# Grupo 4
# Sesión 24 de febrero
#-------------------------------

# Markowitz (1952) - Modelo Media Varianza (MV)
# Algoritmo de optimizacion - Formulación general del problema de optimización

# Carga de librerias
library(xts)
library(zoo)
library(TTR)
library(quantmod)

rm(list=ls()) #Limpia el Environment de R

# 0) INSUMOS DEL MODELO Y PARAMETRIZACIÓN

# Carga de datos
activos <- c("TSLA","BAC","MA","KO","F") # Se pueden ingresar n activos
fecha1 <- "2011-11-30"
fecha2 <- "2022-01-31"
periodicidad <- "monthly" # También puede ser: daily, weekly

#Creación de precios a partir de los activos definidos en "activos"
precios <- xts()
for(i in 1:length(activos)){
  aux <- Ad(getSymbols(activos[i],from=fecha1,to=fecha2,  #La función Ad solo toma Precio ajustado
                       periodicity=periodicidad,auto.assign=FALSE))
  aux <- na.approx(aux,na.rm=FALSE) # Interpolacion de datos con NA
  precios <- cbind(precios,aux)
}

colnames(precios) <- activos # Pega encabezado de nombres en "precios" - fila título
tclass(precios) <- "Date"  # Formato de fecha en "precios" - columna título

#plot(precios[,1:3], type = "s", main = "Precios", lwd = 4)

# 1) GENERACIÓN DE VARIABLES EXOGENAS Y ALGEBRA DEL MODELO

# a) Modelo MV
retornos <- diff(log(precios))[-1] #[-1]: na.omit
cov <- cov(retornos) # Matriz Var-Cov; para anualizar, operar cov*12
mu <- colMeans(retornos) # Retorno esperado para cada activo

var <- diag(cov)      # Varianza
sigma <- sqrt(var)    # Desviacion estandar 

# b) Construcción del conjunto de portafolios optimos
n <- length(mu)
ones <- rep(1,n)
x <- t(mu)%*%solve(cov)%*%mu   # solve: Genera la matriz inversa
y <- t(mu)%*%solve(cov)%*%ones
z <- t(ones)%*%solve(cov)%*%ones
d <- x*z - y^2

g <- (solve(cov,ones)%*%x-solve(cov,mu)%*%y)%*%solve(d)
h <- (solve(cov,mu)%*%z-solve(cov,ones)%*%y)%*%solve(d)

# 2) FRONTERA EFICIENTE - FE Y PORTAFOLIO DE MÍNIMA VARIANZA
#    wi= g + h*E(rp)

# a) Construcción de la Frontera Eficiente
nport <- 1000
Rp <- seq(min(mu),max(mu),length=nport) # Retorno de los portafolios
wpo <- matrix(0,ncol=n,nrow=nport) # Matriz de pesos - Variables endogenas
sigmapo <- matrix(0,nrow=nport) # Riesgo de los portafolios (varianza)
rpo <- matrix(0,nrow=nport) # Retorno de los portafolios - Verificación alternativa de RP (Opcional)

for(i in 1:nport){
  wi <- g + h * Rp[i]
  sigmapo[i] <- sqrt(t(wi)%*%cov%*%wi)
  rpo[i] <- t(wi)%*%mu
  wpo[i,] <- t(wi)
}

# b) Portafolio de Mínima Varianza Global - PMVG
wpmv <- solve(cov,ones)%*%(1/z)
rpmv <- t(wpmv)%*%mu
sigmapmv <- sqrt(t(wpmv)%*%cov%*%wpmv)

rownames(wpmv) <- activos
barplot(t(wpmv),main="Pesos PMVG",col="orange")

# c) Portafolio con Rp Objetivo (Variable endogena) - P Target
Rp1 <- 0.025
wp1 <- g+h*Rp1
sigmap1 <- sqrt(t(wp1)%*%cov%*%wp1)

rownames(wp1) <- activos
barplot(t(wp1),main="Pesos Rp Target ",col="blue")

# d) Gráfica de FE con 2 portafolios

#quartz()
plot(sigma,mu,main="Plano Riesgo-Retorno",xlim=c(0,max(sigma*1.1)),ylim=c(0,max(mu)),col="red")
lines(sigmapo,rpo,col="blue", lwd = 4)
points(sigmapmv,rpmv,lwd = 4,col="red")
text(sigma,mu,labels=activos,pos=4,cex=0.8)
text(sigmapmv,rpmv,labels="PMVGlob",pos=2)
points(sigmap1,Rp1,lwd = 4,col="red") # Rp1 Target
text(sigmap1,Rp1,labels="P1",pos=2)

# 3) MODELO DE SHARPE (1964) - PORTAFOLIO TANGENTE (Con activo rf)

# a) Portafolio Tangente - Versión zi - PT

rf <- 0 # Tasa libre de riesgo
er <- mu-rf # Exceso de retornos

zi <- solve(cov,er)
wpt <- zi/sum(zi)

rpt <- t(wpt)%*%mu
sigmapt <- sqrt(t(wpt)%*%cov%*%wpt)

#quartz()
barplot(t(wpt),main="Pesos Portafolio Tangente",col="pink1")

# b) Portafolio Tangente - Versión (Extracción de PT de la FE)
# Calcula el Coeficiente de Sharpe para todos los portafolios de la FE

sharpe.port <- (rpo-rf)/sigmapo

#quartz()
plot(sharpe.port)

tabla <- cbind(sharpe.port,wpo)
sort.tabla <- tabla[order(-tabla[,1]),]

port.maxsharpe <- sort.tabla[1,]

wpt2 <- port.maxsharpe[2:length(port.maxsharpe)] # Aproximación

# c) Gráfica de FE 3 portafolios

plot(sigma,mu,main="Plano Riesgo-Retorno",xlim=c(0,max(sigma*1.1)),ylim=c(0,max(mu)),col="red")
lines(sigmapo,rpo,col="blue", lwd = 4)
points(sigmapmv,rpmv,lwd = 4,col="red")
text(sigma,mu,labels=activos,pos=4,cex=0.8)
text(sigmapmv,rpmv,labels="PMVGlob",pos=2)
points(sigmap1,Rp1,lwd = 4,col="yellow") # Rp1 Target
text(sigmap1,Rp1,labels="P1",pos=2)
points(sigmapt,rpt, lwd=4, col="purple")
text(sigmapt,rpt,labels = "PT",pos = 2)

# d) Construcción de la Línea del Mercado de Capitales - LMC

wpc <- seq(0,1.5,length=100)
rpc <- matrix(0,nrow=100)
sigmapc <- matrix(0,nrow=100)  

for(i in 1:length(wpc)){
  rpc[i] <- wpc[i]*rpt+(1-wpc[i])*rf
  sigmapc[i] <- wpc[i]*sigmapt
}

#quartz()
plot(sigma,mu,main="Plano riesgo retorno",xlim=c(0,max(sigma)), ylim=c(0,max(mu*1.3)), col="red")
lines(sigmapo,rpo,col="blue", lwd=4)
points(sigmapmv,rpmv, lwd=4, col="red")
text(sigmapmv,rpmv,labels = "PMVGlob",pos = 2)
text(sigma,mu,labels = activos,pos = 4,cex=0.8)
points(sigmap1,Rp1,lwd = 4,col="yellow") # Rp1 Target
text(sigmap1,Rp1,labels="P1",pos=2)
points(sigmapt,rpt, lwd=4, col="purple")
text(sigmapt,rpt,labels = "PT",pos = 2)
lines(sigmapc,rpc, col="purple")
text(sigmapt,rpt,labels = "PT",pos = 2)
lines(sigmapc,rpc, col="purple")

# 4) LIBRERIA QUADRATIC PROGRAMMING - QUADPROG: UTILIZACIÓN DEL OPTIMMIZADOR
# Variante de solución para optimización

# a) Para Pmv sin restricción wi=1
# PMV

library(quadprog)

Dmat <- cov*2
dvec <- rep(0,n)
Amat<- cbind(mu,ones)
bvec <- c(rpmv,1)
res <- solve.QP(Dmat, dvec, Amat, bvec, meq=2)



wqp <- t(res[["solution"]])
sum(wqp)
colnames(wqp) <- activos
barplot(wqp,main="Pesos Pmv sin Restricciones",col="Green1")

# b) Para Pmv con restricción wi=1
# PMV

library(quadprog) 
Dmat <- cov*2
dvec <- rep(0,n)
Amat<- cbind(mu,ones,diag(1,n))
bvec <- c(rpmv,1,rep(0,n))  # bvec <- c(0.012,1,rep(0,n))
res2 <- solve.QP(Dmat, dvec, Amat, bvec, meq=2)

wqp1 <- t(res2[["solution"]]) # Extrae de la lista de optimización el resultado del portafolio

sum(wqp1)

colnames(wqp1) <- activos

par(mfrow=c(1,2))
barplot(wqp,main="Pesos PMV sin Restricciones",col="Green1",cex.names = 0.7)
barplot(wqp1,main="Pesos PMV con Restricciones",col="yellow",cex.names = 0.7)

# c) Para PT sin restricción wi=1
# PT



library(quadprog)
Dmat <- cov*2
dvec <- rep(0,n)
Amat<- cbind(mu,ones)
bvec <- c(rpt,1)
res3 <- solve.QP(Dmat, dvec, Amat, bvec, meq=2)

wqp2 <- t(res3[["solution"]])

colnames(wqp2) <- activos

# d) Para PT con restricción wi=1
# PT

library(quadprog) 
Dmat <- cov*2
dvec <- rep(0,n)
Amat<- cbind(mu,ones,diag(1,n))
bvec <- c(Rp1,1,rep(0,n))
res4 <- solve.QP(Dmat, dvec, Amat, bvec, meq=2)

wqp3 <- t(res4[["solution"]]) # Extrae de la lista de optimización el rsultado del portafolio

colnames(wqp3) <- activos
Rp1
par(mfrow=c(1,2))
barplot(wqp2,main="Pesos PT sin Restricciones",col="grey",cex.names = 0.7)
barplot(wqp3,main="Pesos PT con Restricciones",col="purple",cex.names = 0.7)

## PREGUNTA: APLIQUE LA LIBRERIA QUADPROG PARA MOSTRAR EL RESULTADO PARA EL PORTAFOLIO TARGET CON
## Y SIN RESTRICCIÓN

# Resumen 4 gráficos
# quartz()
x11()
par(mfrow=c(2,2))
barplot(wqp,main="Pesos PMV",col="Green1",cex.names = 0.7)
barplot(t(wp1),main="Pesos Rp Target ",col="blue",cex.names = 0.7)
barplot(wqp2,main="Pesos PT sin Restricciones",col="grey",cex.names = 0.7)
barplot(wqp3,main="Pesos PT con Restricciones",col="purple",cex.names = 0.7)

# 6) DESEMPEÑO DE PORTAFOLIOS
# Performance

# Evaluacion dentro de muestra: con la misma informacion
# Portafolio T (sin restricción)

# Inversión
valor <- 100

# a) Retornos historicos del portafolio PT (sin restricción)
rpsharpe <- retornos%*%as.numeric(wqp2)
rpsharpe2 <- cbind(retornos,rpsharpe) 

#quartz()
## charts.PerformanceSummary(rpsharpe2[,6]) ## portfolio analytics

# Efecto acumulado en el portafolio
t <- length(rpsharpe)
port.sharpe <- matrix(0,nrow=t)
port.sharpe[1] <- valor

for(i in 2:t){
  port.sharpe[i] <- port.sharpe[i-1]*exp(rpsharpe[i-1])
}

# b) Portafolio PT (con restricción): solo pesos positivos

rpsharpesc <- retornos%*%as.numeric(wqp3)
port.sharpesc <- matrix(0,nrow=t)
port.sharpesc[1] <- valor

for(i in 2:t){
  port.sharpesc[i] <- port.sharpesc[i-1]*exp(rpsharpesc[i-1])
}

# c) Portafolio PMV

rpminv <- retornos%*%wpmv
port.minv <- matrix(0,nrow=t)
port.minv[1] <- valor

for(i in 2:t){
  port.minv[i] <- port.minv[i-1]*exp(rpminv[i-1])
}

# c) Portafolio P Target

rptarget <- retornos%*%as.numeric(wp1)
port.target <- matrix(0,nrow=t)
port.target[1] <- valor

for(i in 2:t){
  port.target[i] <- port.target[i-1]*exp(rptarget[i-1])
}

## PREGUNTA: CONSTRUYA LA PRUEBA DE DESEMPEÑO PARA EL PORTAFOLIO TARGET CON Y SIN CORTOS

# d) Indice de referencia: S&P 500
#    S&P 500: ^GSPC

indice <- "^GSPC"

p.indice <- xts()
for(i in 1:length(indice)){
  aux <- Ad(getSymbols(indice[i],from=fecha1,to=fecha2,
                       periodicity=periodicidad,auto.assign=FALSE))
  aux <- na.approx(aux,na.rm=FALSE) # Interpolaci?n de datos con NA
  p.indice <- cbind(p.indice,aux)
}
colnames(p.indice) <- indice
tclass(p.indice) <- "Date"

r.indice <- diff(log(p.indice))[-1]

port.indice <- matrix(0,nrow=t)
port.indice[1] <- valor

for(i in 2:t){
  port.indice[i] <- port.indice[i-1]*exp(r.indice[i-1])
}

# Gráfica de desempeño (historico)
port.sharpe <- ts(port.sharpe,start=2011,frequency = 12)
port.sharpesc <- ts(port.sharpesc,start=2011,frequency = 12)
port.minv <- ts(port.minv,start=2011,frequency = 12)
port.indice <- ts(port.indice,start=2011,frequency = 12)
port.target <- ts(port.target,start=2011,frequency = 12)

# x11()
plot(port.indice,type="s",main="Grafico de desempeño",ylab="Inversión USD",ylim=c(0,max(port.indice*8)))
legend("topleft",c("Indice","PMV","PT Cortos","PT sin Cortos","PTarget"),
       fill=c("black","red","green","blue","purple"))
lines(port.minv,col="red",type="s")
lines(port.sharpe,col="green",type="s")
lines(port.sharpesc,col="blue",type="s")
lines(port.target,col="purple",type="s")

# Tabla resumen

resumen <- matrix(0,5,3)
retorno.p <- rbind(mean(rpsharpe),mean(rpsharpesc),mean(rpminv),mean(r.indice),mean(rptarget))
riesgo.p <- rbind(sd(rpsharpe),sd(rpsharpesc),sd(rpminv),sd(r.indice),sd(rptarget))
sharpe.p <- rbind((mean(rpsharpe)-rf)/sd(rpsharpe),
                  (mean(rpsharpesc)-rf)/sd(rpsharpesc),
                  (mean(rpminv)-rf)/sd(rpminv),
                  (mean(r.indice)-rf)/sd(r.indice),
                  (mean(rptarget)-rf)/sd(rptarget)) # Sharpe= (Rp-rf)/sigmaP

resumen <- cbind(retorno.p,riesgo.p,sharpe.p)
colnames(resumen) <- c("Retorno","Riesgo","Sharpe")
rownames(resumen) <- c("T con cortos","T sin cortos","PMV","Indice","PTarget")
resumen

#-----------------------------------------------------
