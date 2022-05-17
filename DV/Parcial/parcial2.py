import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style()

appl = yf.download('AAPL', start='2021-01-01', end='2022-04-01')
appl.index = pd.to_datetime(appl.index)

t = appl.shape[0]
alpha = np.nanmean(np.log(appl['Adj Close']/appl['Adj Close'].shift(1)))
sigma = np.std(appl.iloc[0]['Adj Close'])
ruido = np.random.normal(0,1,t)

ln_St = appl.iloc[0]['Adj Close'] * np.exp( alpha * range(t) + sigma*ruido)

legend1 = ''.join((r'$S_{t}$'))
legend2 = ''.join((r'$S_{t} = S_{0}e^{\alpha t + \sigma ruido}$'))

fig = plt.figure()
sns.lineplot(appl.index,appl['Adj Close'], label=legend1)
sns.lineplot(appl.index, ln_St, color = 'salmon', label=legend2)
plt.fill_between(appl.index,appl['Adj Close'], ln_St, color='skyblue',alpha=.8)
plt.title('Acciones de APPLE')
plt.legend(loc='upper left')
plt.show()
