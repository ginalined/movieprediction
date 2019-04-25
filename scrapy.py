# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:00:45 2019

@author: elina
"""
import tmdbsimple as tmdb
import omdb
import time
tmdb.API_KEY ='3b9a3e529c7766103068420ff535794a'
import pandas as pd
from omdb import OMDBClient
#links=pd.read_csv('Movie-Recommender-System---SVD-CUR-Collaborative-Filtering-master/links.csv')
tmdbid=links['tmdbId']
client = OMDBClient(apikey='62c7b341')
omdb.set_default('apikey', '62c7b341')

# =============================================================================
# for x in range(20):
#     try:
#         count=0
#         movie = tmdb.Movies(tmdbid[x])
#         print(movie)
#         response = movie.info()
#         links.loc[x,'tbid']=response['vote_average']
#         t=response['imdb_id']
#         print('get')
#         time.sleep(0.25)
#     # =============================================================================
#         imdb=omdb.imdbid(t)
#     # =============================================================================
#         links.loc[x,'imid']=imdb['imdb_rating']
#     except:
#         time.sleep(2)
#         print('miss')
#         count+=1
#         x-=1
#         if count==3:
#             x+=1
# =============================================================================


print(links)
outfile = open('all_rating.csv', 'wb')
links.to_csv('all_rating.csv')
outfile.close()
        

