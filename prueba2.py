import pandas as pd
import numpy as np

data = pd.read_csv('D:/Downloads/owid-covid-data.csv')
print(data.columns)
data = data[['date','location','people_fully_vaccinated_per_hundred','new_cases_per_million','new_deaths_per_million']]
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date',drop=True)


data = data[(data['location']=='Argentina')|
            (data['location']=='Bolivia')|
            (data['location']=='Brazil')|
            (data['location']=='Chile')|
            (data['location']=='Colombia')|
            (data['location']=='Costa Rica')|
            (data['location']=='Ecuador')|
            (data['location']=='Mexico')|
            (data['location']=='Paraguay')|
            (data['location']=='Peru')|
            (data['location']=='Uruguay')
            ]

countries = ['Argentina','Bolivia','Brazil','Chile','Colombia','Costa Rica','Ecuador',
             'Mexico','Paraguay','Peru','Uruguay']

vaccines = pd.DataFrame({})
for c in countries:
    df_temp = data.groupby('location').get_group(c)
    df_temp = df_temp['people_fully_vaccinated_per_hundred'].resample('M').last().apply(lambda x: float(x))

    vaccines = pd.concat([vaccines,df_temp],axis=1)
vaccines.columns = countries

cases = pd.DataFrame({})
for c in countries:
    df_temp = data.groupby('location').get_group(c)
    df_temp = df_temp['new_cases_per_million'].resample('M').mean().apply(lambda x: float(x))

    cases = pd.concat([cases,df_temp],axis=1)
cases.columns = countries

deaths = pd.DataFrame({})
for c in countries:
    df_temp = data.groupby('location').get_group(c)
    df_temp = df_temp['new_deaths_per_million'].resample('M').mean().apply(lambda x: float(x))

    deaths = pd.concat([deaths,df_temp],axis=1)
deaths.columns = countries

# vaccines.to_excel('D:/Desktop/vaccines.xlsx')
# cases.to_excel('D:/Desktop/cases.xlsx')
deaths.to_excel('D:/Desktop/deaths.xlsx')
