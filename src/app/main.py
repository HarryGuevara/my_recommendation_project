import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
import requests
from io import StringIO

# URL base de tu repositorio GitHub
BASE_URL = "https://raw.githubusercontent.com/HarryGuevara/my_recommendation_project/main/data/CSV/"

# Función para cargar los CSV desde GitHub con manejo de errores
def load_csv_from_github(filename: str, usecols=None, sample_frac=1.0) -> pd.DataFrame:
    url = BASE_URL + filename
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza un error si no se puede cargar el archivo
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, usecols=usecols)
        if sample_frac < 1.0:  # Reducir filas si se especifica
            df = df.sample(frac=sample_frac, random_state=42)
        return df
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error cargando archivo {filename}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error procesando archivo {filename}: {e}")

# Cargar los datos con manejo de errores
try:
    cast_df = load_csv_from_github(
        'cast_desanidado.csv',
        usecols=['actor_id', 'actor_name', 'movie_id'],
        sample_frac=0.5
    )
    crew_df = load_csv_from_github(
        'crew_desanidado.csv',
        usecols=['crew_id', 'job', 'movie_id', 'director_name'],
        sample_frac=0.5
    )
    movie_df = load_csv_from_github(
        'movies_datasetc.csv',
        usecols=['movie_id', 'title', 'release_date', 'vote_average', 'vote_count', 'revenue', 'budget'],
        sample_frac=0.5
    )
except RuntimeError as e:
    print(e)
    cast_df, crew_df, movie_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Inicializar la aplicación FastAPI
app = FastAPI()

# Endpoints
@app.get("/cantidad_filmaciones_mes/{mes}")
async def cantidad_filmaciones_mes(mes: str):
    if movie_df.empty:
        raise HTTPException(status_code=500, detail="Datos no cargados correctamente")
    mes_df = movie_df[movie_df['release_date'].str.contains(mes, case=False, na=False)]
    return {"mes": mes, "cantidad": len(mes_df)}

@app.get("/cantidad_filmaciones_dia/{dia}")
async def cantidad_filmaciones_dia(dia: str):
    if movie_df.empty:
        raise HTTPException(status_code=500, detail="Datos no cargados correctamente")
    dia_df = movie_df[movie_df['release_date'].str.contains(dia, case=False, na=False)]
    return {"dia": dia, "cantidad": len(dia_df)}

@app.get("/score_titulo/{titulo_de_la_filmacion}")
async def score_titulo(titulo_de_la_filmacion: str):
    if movie_df.empty:
        raise HTTPException(status_code=500, detail="Datos no cargados correctamente")
    pelicula = movie_df[movie_df['title'].str.contains(titulo_de_la_filmacion, case=False, na=False)]
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    return {
        "titulo": pelicula.iloc[0]['title'],
        "año_estreno": pelicula.iloc[0]['release_date'][:4],
        "score": pelicula.iloc[0]['vote_average']
    }

@app.get("/votos_titulo/{titulo_de_la_filmacion}")
async def votos_titulo(titulo_de_la_filmacion: str):
    if movie_df.empty:
        raise HTTPException(status_code=500, detail="Datos no cargados correctamente")
    pelicula = movie_df[movie_df['title'].str.contains(titulo_de_la_filmacion, case=False, na=False)]
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    votos = pelicula.iloc[0]['vote_count']
    if votos > 2000:
        return {
            "titulo": pelicula.iloc[0]['title'],
            "cantidad_votos": votos,
            "promedio_voto": pelicula.iloc[0]['vote_average']
        }
    else:
        return {"mensaje": "La película tiene menos de 2000 votos."}

@app.get("/get_actor/{nombre_actor}")
async def get_actor(nombre_actor: str):
    if cast_df.empty or movie_df.empty:
        raise HTTPException(status_code=500, detail="Datos no cargados correctamente")
    actor_df = cast_df[cast_df['actor_name'].str.contains(nombre_actor, case=False, na=False)]
    if actor_df.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado")
    peliculas_actor = actor_df.merge(movie_df, left_on='movie_id', right_on='id')
    exito_promedio = peliculas_actor['revenue'].mean()
    cantidad_peliculas = len(peliculas_actor)
    return {
        "actor": nombre_actor,
        "exito_promedio": exito_promedio,
        "cantidad_peliculas": cantidad_peliculas
    }

@app.get("/get_director/{nombre_director}")
async def get_director(nombre_director: str):
    if crew_df.empty or movie_df.empty:
        raise HTTPException(status_code=500, detail="Datos no cargados correctamente")
    director_df = crew_df[crew_df['director_name'].str.contains(nombre_director, case=False, na=False)]
    if director_df.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado")
    peliculas_director = director_df.merge(movie_df, left_on='movie_id', right_on='id')
    peliculas_info = peliculas_director[['title', 'release_date', 'revenue', 'budget']].copy()
    peliculas_info['ganancia'] = peliculas_info['revenue'] - peliculas_info['budget']
    exito_promedio = peliculas_info['ganancia'].mean()
    return {
        "director": nombre_director,
        "exito_promedio": exito_promedio,
        "peliculas": peliculas_info.to_dict(orient='records')
    }
