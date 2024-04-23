import pandas as pd
import numpy as np

def create_csv_rond():
    villes = ["Paris", "Londres", "Berlin", "Madrid", "Rome", "Varsovie", "Budapest", "Vienne", "Prague", "Bucarest"]
    sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    #create random coordinates in longitude and latitude
    longitudes = np.random.uniform(-180, 180, len(villes))
    latitudes = np.random.uniform(-90, 90, len(villes))
    #create a dataframe
    df = pd.DataFrame({"villeID":villes, "size":sizes, "lat":latitudes, "long":longitudes})
    #save the dataframe to a csv file
    df.to_csv("rond.csv", index=False)

def create_csv_carre():
    villes = ["Paris", "Londres", "Berlin", "Madrid", "Rome", "Varsovie", "Budapest", "Vienne", "Prague", "Bucarest"]
    sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    #create random coordinates in longitude and latitude
    X = np.random.uniform(-180, 180, len(villes))
    Y = np.random.uniform(-180, 180, len(villes))
    #create a dataframe
    df = pd.DataFrame({"villeID":villes, "size":sizes, "Xi":X, "Yi":Y})
    #save the dataframe to a csv file
    df.to_csv("carre.csv", index=False)

create_csv_rond()
create_csv_carre()
