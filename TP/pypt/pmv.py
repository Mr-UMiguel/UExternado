import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

import cvxopt
from cvxopt import matrix
from cvxopt import solvers

sns.set_theme(font="Times New Roman",font_scale=1)

def pmv_plot(mu,vcov,g,h,
            sigmapmv,rpmv,
            lmc=False,include_assets=False, sigma=None,symbols=None):
    """
    Función auxiliar que permite graficar la frontera eficiente
    """

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

    if include_assets == True:  
        if any(sigma) != None:
            mu_norm = np.concatenate((mu,rpo))
            mu_norm = normalize(mu_norm, min(mu_norm), max(mu_norm))
            sigma_norm = np.concatenate((sigma,sigmapo))
            sigma_norm = normalize(sigma_norm, min(sigma_norm), max(sigma_norm))
            
            xmin, xmax = min(sigma_norm)-(1/len(sigma_norm)) , max(sigma_norm)+(1/len(sigma_norm))
            ymin, ymax = min(mu_norm)-(1/len(mu_norm)), max(mu_norm)+(1/len(mu_norm))
    else:
        xmin, xmax = sigmapo.min()-(1/len(sigmapo)), sigmapo.max()+(1/len(sigmapo))
        ymin, ymax = rpo.min()-(1/len(rpo)), rpo.max()+(1/len(rpo))

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

    sns.scatterplot(x=sigmapmv,y=rpmv, color="#E3550E", marker="$\circ$", ec="face", s=200, label=pmvglegend)
    
    if include_assets == True:
        sns.scatterplot(x=sigma_norm[:len(sigma)], y=mu_norm[:len(mu)],color="#EAECE7", marker="$\circ$", ec="face", s=80)
        for mn, sm, name in zip(mu_norm[:len(mu)],sigma_norm[:len(sigma)],symbols):
            plt.annotate(name,xy=(sm,mn))
        
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

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def solvers_qp(vcov,mu,optimal_portfolio,inequality=False,):
    """
    Definimos la función solvers_qp que utiliza el solvers.qp del módulo cvxopt

    Optimiza un problema de la forma 

    min (1/2)x'Px + q'x
    s.a Gx <= h
        Ax = b

    Si adaptamos esta optimización a nuestro ejercicio de optimización del
    portafolio óptimo entonces tenemos que 
    
    Si inequality = False

        min L = (1/2)w'Pw
        s.a w'E = E(rp)
            w'1 = 1
    Si inequality = True

        min L = (1/2)w'Pw
        s.a w'E = E(rp)
            w'1 >= 0
    
    Es decir, podemos restringir la optimización para que la suma de las ponderaciones
    sea uno o bien, para que sea no negativa 

    Parámetros:
    -------------------------------------------------------------------

    vcov : Matriz (n x n) de varianzas y covarianzas


    mu : vector de tamaño (n,) de los retornos esperados de los n activos

    optimal_portfolio : Valor del retorno óptimo, este debe estar dentro de la frontera eficiente
                        puede ser el retorno del portafolio de mínima varianza global, el portafolio
                        óptimo de sharpe, o cualquier otro. 

    inequality : bool , default = False si w'1 = 1 o True si w'1 >= 0

    Ejemplo: 
    -------------------------------------------------------------------
    vcov : (5x5)

        [[0.02450049 0.00388752 0.00228452 0.00095474 0.00308689]
        [0.00388752 0.00874613 0.00182295 0.00094029 0.00445079]
        [0.00228452 0.00182295 0.00385652 0.00142886 0.00119709]
        [0.00095474 0.00094029 0.00142886 0.0021134  0.00087716]
        [0.00308689 0.00445079 0.00119709 0.00087716 0.00734027]]
    mu : (5,)
    
        [0.04029284 0.01008825 0.02163079 0.00754402 0.00529053]

    optimal_portfolio : rpmv

        [0.01000976]

    inequality : False

    [in] solv = solvers_qp(vcov = vcov, mu = mu, optimal_portfolio=rpt, inequality=True)
        print(solv)

    [out]
        [ 1.26e-02]
        [ 3.92e-02]
        [ 1.56e-01]
        [ 6.81e-01]
        [ 1.10e-01] 
    """
    n = len(mu)
    if inequality == False:
        P = matrix(vcov)
        q = matrix(np.zeros((n,1)))
        G = matrix(np.concatenate((
            -np.transpose(np.array(mu)).reshape((n,1)),
            -np.ones(n).reshape(n,1)),1).T)
        h = matrix(-np.array([optimal_portfolio,[1]]))
    elif inequality == True:
        P = matrix(vcov)
        q = matrix(np.zeros((n,1)))
        G = matrix(np.concatenate((
            -np.transpose(np.array(mu)).reshape((n,1)),
            -np.ones(n).reshape(n,1),
            -np.diag(np.full(n,1))),1).T)
        h = matrix(-np.concatenate((
            np.array([optimal_portfolio,[1]]),
            np.zeros(n).reshape(n,1)),0))

    response = np.array(solvers.qp(P=P,q=q,G=G,h=h,show_progress=False)['x']).reshape(-1)
    return response

class PMV:

    def __init__(self,returns):
        """
        Miguel Angel Manrique Rodriguez
        
        Esta clase le permite calcular el portafolio óptimo de mínima varianza global PMVg,
        el portafolio óptimo de sharpe o portafolio tangente de sharpe, y la curva de mercado
        de capitales.

        Es un ejercicio educativo para la materia Teoría de Portafolios de la Universidad Externado de Colombia
        a cargo del profesor Oscar Reyes

        Parámetros
        ------------------------------------------------

        returns: pd.core.frame.Dataframe object

                Data frame de pandas con los retornos de los precios de los activos
                para calcular el retorno puede usar get_data().returns(), use
                help(get_data.precios)
        """
        self.returns = returns

    def mv(self,target_return=None):
        """
        El método mv de Mínima Varianza le permite calcular el portafolio óptimo de mínima varianza global
        y el portafolio óptimo con rentorno objetivo deseados


        Parámetros
        ---------------------

        target_return : float

            Si desea calcular el portafolio óptimo para un retorn objetivo deseado
            utilice target_return y especifique el retorno deseado


        Valores devueltos (atributos)
        ----------------------------------------------------------------
        sea n el número de acciones

            vcov : matriz (n x n) de varianzas y covarianzas 
            corr : matriz (n x n) de coeficientes de correlación de pearson 
            mu   : vector (n,) de retornos promedio 
            var  : vector (n,) de varianzas 
            sigma: vector (n,) de desviaciónes estándar

            rpmv : retorno (rentabilidad) del portafolio de mínima varianza global
            sigmapmv : varianza (riesgo) del portafolio de mínima varianza global
            wpmv : ponderaciónes del portafolio de mínima varianza global

            pmv : rpmv, sigmapmv, wpmv

        Si target_return = True:
            rp_target : target_return
            sigma_target : varianza (riesgo) del portafolio óptimo con retorno objetivo
            wp_target : ponderaciónes del portafolio óptimo con retorno objetivo
        Métodos
        ----------------------------------------------------------------
        plot() : method

            Grafica la frontea eficiente de markowitz con el portafolio óptimo

            include_assets : bool, default=False

                Si True entonces se estandarizan la media y la desviación estandar y se incluyen
                todos los activos de returns, es un parámetro puramente estético.

                Si False no se estandariza la media y la desviación estandar y no se incluyen
                los activos de returns 
        """
        # variance-covariance matrix and correlation matrix
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

        # longitud del vector mu (cantidad de activos)
        n = len(self.mu)

        # Creamos un vector de unos de longitud n
        ones = np.ones(n)

        # Solucionamos el ejercicio de optimización (ver notas.ipynb)
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
        ## target_return
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

        def plot(include_assets=False):
            fig = pmv_plot(
                mu = self.mu,
                vcov = self.vcov,
                g =  self.g,
                h =  self.h,
                sigmapmv = self.sigmapmv,
                rpmv = self.rpmv,
                include_assets = include_assets,
                sigma = self.sigma,
                symbols = self.returns.columns
            )
            sns.scatterplot(x=self.sigma_target,y=self.rp_target, color="#BE33FF", marker="$\circ$", ec="face", s=150, label=ptlegend)
            plt.title("Portafolo óptimo de Mínima Varianza - MV")
            plt.show()

        self.plot = plot

        return self




        # return self

    def sharpe(self):
        """
        El método sharpe de Sharpe le permite calcular el portafolio óptimo de sharpe, o portafolio 
        tangente de sharpe y la linea del mercado de capitales



        Valores devueltos (atributos)
        ----------------------------------------------------------------
        sea n el número de acciones

            rpt : retorno (rentabilidad) del portafolio de sharpe
            sigmapt : varianza (riesgo) del portafolio de sharpe
            wpt : ponderaciónes del portafolio de sharpe

            psharpe = rpt, sigmapt, wpt


        Métodos
        ----------------------------------------------------------------
        plot() : method

            Grafica la frontea eficiente de markowitz con el portafolio óptimo

            include_assets : bool, default=False

                Si True entonces se estandarizan la media y la desviación estandar y se incluyen
                todos los activos de returns, es un parámetro puramente estético.

                Si False no se estandariza la media y la desviación estandar y no se incluyen
                los activos de returns 

            lmc : bool, default=False

                Si True entonces calcula la linea del mercado de capitales con activo libre de riesgo
                de lo contrario False
        """
        mv = self.mv()
        ########################################################################
        ## Introducción del activo libre de riesgo
        rf = 0
        er = mv.mu - rf
        zi = np.linalg.solve(mv.vcov,er)


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

        

        def plot(lmc=False,include_assets=False):
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
                lmc = lmc,
                include_assets = include_assets,
                sigma = self.sigma,
                symbols = self.returns.columns
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

    def sortino(self,rf,cortos):
            """
            El método sortino le permite calcular el portafolio óptimo de sortino

            Valores devueltos (atributos)
            ----------------------------------------------------------------
            sea n el número de acciones

                rps : retorno (rentabilidad) del portafolio de sortino
                sigmaps : varianza (riesgo) del portafolio de sortino
                wps : ponderaciónes del portafolio de sortino

                psortino = rps, sigmaps, wps


            Métodos
            ----------------------------------------------------------------
            plot() : method

                Grafica la frontea eficiente de markowitz con el portafolio óptimo

                include_assets : bool, default=False

                    Si True entonces se estandarizan la media y la desviación estandar y se incluyen
                    todos los activos de returns, es un parámetro puramente estético.

                    Si False no se estandariza la media y la desviación estandar y no se incluyen
                    los activos de returns 

                lmc : bool, default=False

                Si True entonces calcula la linea del mercado de capitales con activo libre de riesgo
                de lo contrario False
            """
            mv = self.mv()
            
            ###############################################
            ### Construcción del portafolio óptimo de sortino
            n = self.returns.shape[1]
            T = self.returns.shape[0]

            mu = mv.mu
            ### E[min{ri-rf,0}]^2
            E_min = pd.DataFrame({})
            for i in range(n):
                E_min[self.returns.columns[i]] =  np.array(np.minimum(self.returns.iloc[:,i]-rf,0)**2)

            ### Sum E[min{ri-rf,0}]^2
            sum_E_min = E_min.agg(np.sum)

            ## Semi variana: (Sum E[min{ri-rf,0}]^2)/T
            sm = sum_E_min/(T-1)

            ## Ratio de sortino: [E(ri)-rf/semivarianza del mercado]
            self.sortino_ratio = (mu - rf)/np.sqrt(sm)

            # Valor esperado
            Er = mu - rf
            # Semivarianza
            Svcov = E_min.cov()

            # Con optimización
            solv = solvers_qp(vcov=np.asmatrix(Svcov), mu = mu, optimal_portfolio  = [np.min(mu)],inequality=cortos)

            #Ponderaciones óptimas
            self.wpst = np.array(solv)
            # Retorno esperado
            self.rpst = np.array([self.wpst @ mu])
            # Riesgo esperado
            self.sigmapst = np.array([np.sqrt(np.transpose(self.wpst)@Svcov@self.wpst)])

            # Sortino Portfolio
            self.psortino = self.rpst, self.sigmapst, self.wpst
            return self

    def treynor(self,index_returns):
            """
            El método treynor le permite calcular el portafolio óptimo de Treynor

            Valores devueltos (atributos)
            ----------------------------------------------------------------
            sea n el número de acciones

                rptr : retorno (rentabilidad) del portafolio de Treynor
                sigmaptr : varianza (riesgo) del portafolio de Treynor
                wptr : ponderaciónes del portafolio de Treynor

                ptreynor = rptr, sigmaptr, wptr


            Métodos
            ----------------------------------------------------------------
            plot() : method

                Grafica la frontea eficiente de markowitz con el portafolio óptimo

                include_assets : bool, default=False

                    Si True entonces se estandarizan la media y la desviación estandar y se incluyen
                    todos los activos de returns, es un parámetro puramente estético.

                    Si False no se estandariza la media y la desviación estandar y no se incluyen
                    los activos de returns 
            """

            index_returns = np.array(index_returns)

            def ols(Y,X):
                n = len(X)

                Y = np.asarray(Y).reshape((n,1))
                X = np.concatenate((
                        np.ones(n).reshape((n,1)),
                        np.asmatrix(X).reshape((n,1))
                    ),axis=1)
                
                xtx = np.transpose(X) @ X
                xtx_x = np.linalg.inv(xtx)
                xy = np.transpose(X)@Y
                b = xtx_x @ xy
                B = np.squeeze(np.asarray(b))

                xb = X * b
                e = Y - xb
                e = np.var(e,ddof=1)
                
                return B[1], e

            mv = self.mv()
            
            ###############################################
            ### Construcción del portafolio óptimo de treynor
            n = self.returns.shape[1]
            betas = np.zeros(n)
            varianza_error = np.zeros(n)

            mu = mv.mu
            vcov = mv.vcov
            sigma = np.sqrt(np.diag(vcov))
            rf = 0

            # regresión iterativa para los parámetros
            for i in range(n):
                betas[i], varianza_error[i] = ols(Y=self.returns.iloc[:,i],X=index_returns)

            ratio_treynor = (mu-rf)/betas

            # Cálculos de los ratios 1 y 2 y las sumas acumuladas

            matriz = pd.DataFrame({
                'Ratio_treynor':ratio_treynor,
                'Betas' : betas,
                'Var_Error': varianza_error,
                'Mu': mu,
                'Sigma': sigma
            },index=self.returns.columns)

            matriz = matriz.sort_values(by=['Ratio_treynor'],ascending=False)

            ratio1 = np.cumsum(((matriz['Mu']-rf)*matriz['Betas'])/(matriz['Var_Error']))
            ratio2 = np.cumsum(((matriz['Betas'])**2)/(matriz['Var_Error']))

            sigma_mkt = np.std(index_returns)
            tasac = ((sigma_mkt**2)*ratio1)/((1+((sigma_mkt**2)*ratio2)))

            diff = matriz['Ratio_treynor'] - tasac
            cond_diff =  diff[diff>0]
            n_optimo = len(cond_diff)
            cmax = max(tasac)

            zi = (matriz['Betas']/matriz['Var_Error']) * (matriz['Ratio_treynor']-cmax)
            zi = np.maximum(zi,0)

            self.wptr = np.array(zi/(sum(zi)))
            self.rptr = np.array(self.wptr@mu)
            self.sigmaptr = np.squeeze([np.asarray(np.sqrt(self.wptr@vcov@self.wptr))])

            self.ptreynor = self.rptr, self.sigmaptr, self.wptr, matriz

            return self



# def plot3d():
#     p1 = pd.Series(np.random.normal(40,10,100))
#     p2 = pd.Series(np.random.normal(100,20,100))

#     r1 = np.log(p1/(p1.shift(1))).dropna()
#     r2 = np.log(p2/(p2.shift(1))).dropna()

#     mu = np.array([np.mean(r1),np.mean(r2)])
#     vcov = np.cov(r1,r2)
#     sigma = np.sqrt(np.diag(vcov))
#     weights = np.random.dirichlet(np.ones(2),size=100)

#     w1 = weights[:,0]
#     w2 = weights[:,1]

#     sigma1 = sigma[0]
#     sigma2 = sigma[1]
#     cov = vcov[0,1]
#     #minimizar la varianza
#     V = (w1**2)*(sigma1**2)+(w2**2)*(sigma2**2)+ (2*w1*w2)* cov

#     xmin, xmax = min(w1) , max(w1)
#     ymin, ymax = min(w2), max(w2)

#     fig = plt.figure(figsize=(10,7))
#     ax = plt.axes(projection='3d')
#     x = np.linspace(xmin,xmax,100)
#     y = np.linspace(ymin,ymax,100)
#     X, Y = np.meshgrid(x, y)
#     Z = (X**2)*(sigma1**2)+(Y**2)*(sigma2**2)+ (2*X*Y)* cov
#     ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                     cmap='viridis', edgecolor='none')
#     ax.set_title('Distribuciones mltivariadas');
#     plt.show()