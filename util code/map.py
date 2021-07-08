import pandas as pd
import folium
from folium import plugins
map = folium.Map(location=[9.0820,8.6753], width='100%', height='100%', left='0%', top='0%', position='relative',zoom_start=6, tiles='Stamenterrain')

df = pd.read_csv("ng.csv")

df_corona = pd.read_csv('NigCovidStats.csv')
df_corona.sort_values(by='state')
df_corona = df_corona.drop(['Unnamed: 0'],  axis=1)
#df_corona.head()
df_corona.style.background_gradient(cmap='Reds')

for lat, lon, value, name in zip(df['lat'], df['lng'], df['population'], df['city']):
    folium.CircleMarker([lat, lon], color='crimson', fill = True, radius=value*0.000005, popup =name).add_to(map)