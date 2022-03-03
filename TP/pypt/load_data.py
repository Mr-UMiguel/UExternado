import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(font="Times New Roman",font_scale=1)

class get_data():

    def __init__(self,symbols,start_date,end_date,frequency=None,type_price="Adj Close"):
        """
        Miguel Angel Manrique Rodriguez
        
        Esta clase le permite descargar datos del precio de los instrumentos financieros presentes en Yahoo Finance

        Es un ejercicio educativo para la materia Teoría de Portafolios de la Universidad Externado de Colombia
        a cargo del profesor Oscar Reyes

        2022-1S

        Parámetros
        ---------------------
        symbols: list

                Lista de símbolos o 'Tickets' usados en yahoo finance como nemotécnico de los diferentes 
                instrumentos financieros, para mayor información ver la librería yfinance

        start_date: str
        
                    Fecha inicial de los datos
                    'YYYY-DD-MM'

        end_date: str
        
                    Fecha Final de los datos
                    'YYYY-DD-MM'

        frequency: str , default = "1d"
        
                    Periodicidad de los datos, puede ser 1d, M, Q, Y ---> ver yfinance

        type_price: str , default = 'Adj Close'

                    Tipo de precio, a saber: 
                    'Close' : Precio de cierre
                    'Adj Close' : Precio de cierre ajustado              
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.type_price = type_price
        pass

    
    def prices(self,plot={'draw':False,'standarise':False}):
        """
        El método prices obtiene los precios de yfinance con los parámetros de la clase get_data

        Adeás se utilizan métodos de la librería pandas y de numpy para la manipulación de los 
        dataframes

        Parámetros
        ---------------------
        plot : dict

            'draw' : bool , default=False

                True si se quiere graficar el precio de  symbols
                False si no se quiere graficar el precio de  symbols

            'standarise' : bool , default=False

                True si se quiere graficar el precio de  symbols estandarizados
                False si no se quiere graficar el precio  de symbols estandarizados

        Ejemplo
        ---------------------
        [in]  get_data(symbols=['AAPL','TSLA'],start_date="2021-11-30",end_date="2022-01-31",frequency='M').prices()

        [out] 
                            AAPL Adj Close  TSLA Adj Close
            2021-11-30      165.089676      1144.760010
            2021-12-31      177.344055      1056.780029
            2022-01-31      170.113266      846.349976

        return: pandas.core.frame.Dataframe
        """
        #Creamos un dataframe vacío para actualizar con nuevos datos
        self.__prices = pd.DataFrame({})
        # Iteramos cada uno de los símbolos para obtener la columna precio o precio ajustado
        for symbol in self.symbols:
            # Descargamos la data usando la api de yahoo finance
            self.__data = yf.download(symbol,start=self.start_date,end=self.end_date,progress=False)
            # Escogemos la columna deseada
            if self.type_price == "Close":
                self.__data = self.__data['Close']
                self.__data.rename(f"{symbol}",inplace=True)
            elif self.type_price == "Adj Close" or self.type_price == None:
                self.__data = self.__data['Adj Close']
                self.__data.rename(f"{symbol}",inplace=True)

            else: raise ValueError("type_price onyl can be 'Close' or 'Adj Close'")

            # Actualizamos el dataframe
            self.__prices = pd.concat([self.__prices,self.__data],axis="columns")
        
        # Si deseamos ver los datos en diferentes frecuencias podemos usar el método .resample() de pandas
        # tomando el último datos observado de la frecuencia corresponiente
        # P.e: si escogemos 'M' tomaremos el precio del último día bursatil del mes
        self.__prices.index = pd.to_datetime(self.__prices.index)
        if self.frequency != None:
            self.__prices = self.__prices.resample(self.frequency).last()
        
        # Graficamos 
        if plot['draw'] == True:
            if plot['standarise'] == True:
                print("Hola")
                self.__prices_plot = (self.__prices-self.__prices.mean())/self.__prices.std()
            else:
                self.__prices_plot = self.__prices
            fig = plt.figure(figsize=(10,7))
            plt.plot(self.__prices_plot)
            plt.title(f"{self.type_price} prices chart from {self.start_date} to {self.end_date}")
            plt.legend(self.symbols,loc="upper left")
            plt.show()

        return self.__prices

    def returns(self,return_type,plot=False):
        """
        El método returns obtiene los retronos de los precios de yfinance con los parámetros 
        de la clase get_data

        Además se utilizan métodos de la librería pandas y de numpy para la manipulación de los 
        dataframes

        Parámetros
        ---------------------
        return_type : str , default='log'

            'log' si se calcula el retorno logarítimco
            'ari' si se calcula el retorno aritmético
        plot : bool, default=False

            True si se quiere graficar los retornos
            False si no se quiere graficar los retornos

        Ejemplo
        ---------------------
        [in]  get_data(symbols=['AAPL','TSLA'],start_date="2021-11-30",end_date="2022-01-31",frequency='M').returns(return_type='log')

        [out] 
                            AAPL Adj Close  TSLA Adj Close
            2021-12-31      0.071603        -0.079968
            2022-01-31      -0.041627       -0.222049

        return: pandas.core.frame.Dataframe
        """
        if return_type == "log":
            self.__ret = np.log(self.prices()/self.prices().shift(1))
        elif return_type == "ari":
            self.__ret = (self.prices()/self.prices().shift(1)) - 1
        else: raise ValueError("Unknown return_type")

        self.__ret = self.__ret.dropna()

        # Graficamos
        if plot == True:
            fig = plt.figure(figsize=(10,7))
            plt.plot(self.__ret)
            plt.title(f"{return_type} returns chart from {self.start_date} to {self.end_date}")
            plt.legend(self.symbols,loc="upper left")
            plt.show()

        return self.__ret

