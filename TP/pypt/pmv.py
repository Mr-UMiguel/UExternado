import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from pyparsing import line
import seaborn as sns

sns.set_theme(font="Times New Roman",font_scale=1)


class PMV:

    def __init__(self,returns):
        self.returns = returns

    def mv(self,target_return=None):
        """
        
        
        Valores devueltos
        ----------------------------------------------------------------

            wpmv : ponderaciónes del portafolio de mínima varianza
            rpmv : retorno (rentabilidad) del portafolio de mínima varianza
            sigmapmv : varianza (riesgo) del portafolio de mínima varianza
        
        """
        # variance-covariance matriz and correlation matrix
        try:
            self.vcov, self.corr = np.matrix(self.returns.cov()), np.matrix(self.returns.corr())
        except:
            raise ValueError("returns must be a pandas data frame")

        # mu es un vector de retornos esperados de cada activo 
        # global mu
        self.mu = np.array(self.returns.agg(np.mean))
        # Varianza y desviación estándar de los activos
        self.var = np.diag(self.vcov)
        self.sigma = np.sqrt(self.var)

        ################################################################
        # Construcción de los portafolios óptimos

        # longitud del vector mu
        n = len(self.mu)

        # Creamos un vector de unos de longitud n
        ones = np.ones(n)

        # Solucionamos el ejercicio de optimización
        x = self.mu @ np.linalg.inv(self.vcov) @ self.mu
        y = self.mu @ np.linalg.inv(self.vcov) @ ones
        z = ones @ np.linalg.inv(self.vcov) @ ones

        d = x*z - y**2
        self.g = (np.linalg.solve(self.vcov,ones) * np.array(x)-np.linalg.solve(self.vcov,self.mu)*np.array(y)) * 1/d
        self.h = (np.linalg.solve(self.vcov,self.mu) * np.array(z)-np.linalg.solve(self.vcov,ones)*np.array(y)) * 1/d


        self.wpmv = np.squeeze(np.asarray(np.linalg.solve(self.vcov,ones) * 1/z))
        self.rpmv = np.asarray(self.wpmv @ self.mu).reshape(-1)
        self.sigmapmv = np.asarray(np.sqrt(self.wpmv@self.vcov@np.transpose(self.wpmv))).reshape(-1)

        self.pmv = self.rpmv, self.sigmapmv, self.wpmv

        ################################################################
        ## Calculamos los pesos óptimos y el riesgo con un retorno Objetivo

        if target_return != None:
            self.rp_target = target_return
            self.wp_target = np.asarray(self.g + (self.h*self.rp_target)).reshape(-1)
            self.sigma_target = np.asarray(np.sqrt(self.wp_target@self.vcov@self.wp_target)).reshape(-1)
            ptlegend = ''.join((
            'PMVg( '
            r'$\mu=%.2f$, ' % (self.rp_target, ),
            r'$\sigma=%.2f$)' % (self.sigma_target, )))
        else:
            self.rp_target = None
            self.wp_target = None
            self.sigma_target = None
            ptlegend = None

        def plot():
            fig = pmv_plot(
                mu = self.mu,
                vcov = self.vcov,
                g =  self.g,
                h =  self.h,
                sigmapmv = self.sigmapmv,
                rpmv = self.rpmv
            )
            sns.scatterplot(x=self.sigma_target,y=self.rp_target, color="#BE33FF", marker="$\circ$", ec="face", s=150, label=ptlegend)
            plt.title("Portafolo óptimo de Mínima Varianza - MV")
            plt.show()

        self.plot = plot

        return self




        # return self

    def sharpe(self,short_sell=True):
        mv = self.mv()
        ########################################################################
        ## Introducción del activo libre de riesgo
        rf = 0
        er = mv.mu - rf
        if short_sell == True:
            zi = np.linalg.solve(mv.vcov,er)
        else:
            ones = np.ones(len(mv.mu))
            zi = ones @ np.linalg.inv(mv.vcov) @ ones

        self.wpt = np.squeeze(np.asarray(zi / sum(zi))) # "portafolio óptimo de sharpe"


        ##### 
        # Rentabilidad y riesgo del portafolio óptimo
        self.rpt = np.asarray(self.wpt @ mv.mu).reshape(-1)
        self.sigmapt = np.asarray(np.sqrt(self.wpt @ mv.vcov @ self.wpt)).reshape(-1)

        self.psharpe = self.rpt, self.sigmapt, self.wpt
        
        ################################
        # Construcción de la línea del mercado de capitales

        wpc = np.linspace(0,1.5,1000) # LMC
        rpc = []
        sigmapc = []

        for i in range(len(wpc)):
            rpc.append((wpc[i]*self.rpt)+(1-wpc[i])*rf)
            sigmapc.append(wpc[i]*self.sigmapt)

        rpc = np.asarray(rpc).reshape(-1)
        sigmapc = np.asarray(sigmapc).reshape(-1)

        

        def plot(lmc=False):
            ptlegend = ''.join((
            'Psharpe( '
            r'$\mu=%.2f$, ' % (self.rpt, ),
            r'$\sigma=%.2f$)' % (self.sigmapt, )))
            fig = pmv_plot(
                mu = mv.mu,
                vcov = mv.vcov,
                g = mv.g,
                h = mv.h,
                sigmapmv= mv.sigmapmv,
                rpmv = mv.rpmv,
                lmc = lmc
            )
            sns.scatterplot(x=self.sigmapt,y=self.rpt, color="#3393FF", marker="$\circ$", ec="face", s=150, label=ptlegend)
            if lmc == True:
                sns.lineplot(x=[i for i in sigmapc if i <= self.sigmapt],y=[i for i in rpc if i <=self.rpt], label="LMC")
                plt.title("Portafolio óptimo de Sharpe y LMC")
            else:
                plt.title("Portafolio óptimo de Sharpe")
            plt.show()

        self.plot = plot

        return self


def pmv_plot(mu,vcov,g,h,
            sigmapmv,rpmv,lmc=False):

    N = 1000
    Rp = np.linspace(start=np.min(mu), stop=np.max(mu),num=N)
    wpo = np.zeros((N,len(mu)))
    sigmapo = np.zeros(N)
    rpo = np.zeros(N)

    for i in range(N):
        wi = g+h*Rp[i]
        sigmapo[i] = np.sqrt(wi@vcov@np.transpose(wi))
        rpo[i] = wi @ mu
        wpo[i] = wi

    xmin, xmax = sigmapo.min()-(1/len(sigmapo)), sigmapo.max()
    ymin, ymax = rpo.min(), rpo.max()

    pmvglegend = ''.join((
    'PMVg( '
    r'$\mu=%.2f$, ' % (rpmv, ),
    r'$\sigma=%.2f$)' % (sigmapmv, )))

    fig = plt.figure(figsize=(10,7))
    sns.scatterplot(x=sigmapo,y=rpo, color="#0EACE3",label=None)
    for i in range(len(sigmapo)):
        if i == 0:
            plt.fill_between(x=[0,sigmapo[i]], y1=rpo[i], y2=rpo[i], color="#77DDFF",alpha=0.1, interpolate=True)
            plt.fill_between(x=[sigmapo[i],xmax], y1=rpo[i], y2=rpo[i], color="#77FFA9",alpha=0.1, interpolate=True)
        else:
            plt.fill_between(x=[0,sigmapo[i]], y1=rpo[i-1],  y2=rpo[i],color="#77DDFF",alpha=0.1, interpolate = True)
            plt.fill_between(x=[sigmapo[i],xmax], y1=rpo[i-1],  y2=rpo[i],color="#77FFA9",alpha=0.1, interpolate = True)

    sns.scatterplot(x=sigmapmv,y=rpmv, color="#E3550E", marker="$\circ$", ec="face", s=150, label=pmvglegend)
    if lmc == True:
        plt.xlim(0,xmax)
        plt.ylim(0,ymax)
    else:    
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
    plt.xlabel("Riesgo")
    plt.ylabel("Rentabilidad")
    plt.legend(loc="upper left")

    # plt.show()
    return fig