from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import unicodedata
import dask.dataframe as dd

app = FastAPI()

# ✅ Carga bajo demanda para reducir consumo de memoria
def load_movies():
    return pd.read_csv(
        "data/top_20000_movies.csv",
        usecols=['movie_id', 'title', 'popularity', 'vote_average', 'release_year', 'revenue', 'budget'],
        dtype={
            'popularity': 'float32',
            'vote_average': 'float32',
            'release_year': 'int16',
            'revenue': 'float32',
            'budget': 'float32',
            'movie_id': 'int32'
        }
    )

def load_cast():
    return pd.read_csv("data/cast.csv", usecols=['movie_id', 'name_actor'])

def load_crew():
    return pd.read_csv("data/crew.csv", usecols=['movie_id', 'name_job', 'job_crew'])

# 🔹 Normalizar nombres para evitar problemas en búsquedas
def normalizar_nombre(nombre):
    nombre = unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('ascii')
    return nombre.lower()

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de recomendación de películas"}

@app.get("/movies")
def get_movies():
    df = load_movies()
    return df.sample(5).to_dict(orient="records")

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    df = load_movies()
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    mes = mes.lower()
    if mes not in meses:
        raise HTTPException(status_code=400, detail="Mes no válido.")

    cantidad = df[df['release_year'] == meses[mes]].shape[0]
    return {"mensaje": f"{cantidad} películas fueron estrenadas en {mes.capitalize()}"}

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    df = load_movies()
    df['title_normalized'] = df['title'].apply(normalizar_nombre)

    pelicula = df[df['title_normalized'] == normalizar_nombre(titulo)]
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")

    return {
        "mensaje": f"La película {pelicula['title'].values[0]} fue estrenada en {pelicula['release_year'].values[0]} con un score de {pelicula['vote_average'].values[0]}"
    }

@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    df = load_movies()
    pelicula = df[df['title'].str.lower() == titulo.lower()]
    
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")

    if 'vote_count' not in pelicula.columns or pelicula['vote_count'].values[0] < 2000:
        return {"mensaje": f"La película {titulo} no tiene al menos 2000 valoraciones."}

    return {
        "mensaje": f"La película {pelicula['title'].values[0]} fue estrenada en {pelicula['release_year'].values[0]} con {pelicula['vote_count'].values[0]} valoraciones y un promedio de {pelicula['vote_average'].values[0]}"
    }

@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    df = load_movies()
    cast_df = load_cast()

    actor_peliculas = cast_df[cast_df['name_actor'].str.lower() == nombre_actor.lower()]
    if actor_peliculas.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado.")

    peliculas_ids = actor_peliculas['movie_id'].unique()
    peliculas_actor = df[df['movie_id'].isin(peliculas_ids)]

    if peliculas_actor.empty:
        raise HTTPException(status_code=404, detail="No se encontraron películas para este actor.")

    return {
        "mensaje": f"El actor {nombre_actor} ha participado en las siguientes películas:",
        "peliculas": peliculas_actor[['title', 'release_year']].to_dict('records')
    }

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    df = load_movies()
    crew_df = load_crew()

    director_peliculas = crew_df[(crew_df['name_job'].str.lower() == nombre_director.lower()) & (crew_df['job_crew'] == 'Director')]
    if director_peliculas.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado.")

    peliculas_ids = director_peliculas['movie_id'].unique()
    peliculas_director = df[df['movie_id'].isin(peliculas_ids)]

    retorno_total = peliculas_director['revenue'].sum() / peliculas_director['budget'].sum()

    return {
        "mensaje": f"El director {nombre_director} ha conseguido un retorno total de {retorno_total}",
        "peliculas": peliculas_director[['title', 'release_year', 'revenue', 'budget']].to_dict('records')
    }

# 🔹 Recomendaciones con cálculo bajo demanda
def compute_similarity():
    df = load_movies()

    df['return'] = np.where(
        df['budget'] != 0,
        df['revenue'] / df['budget'],
        0
    )

    features = df[['vote_average', 'popularity', 'release_year', 'return']]
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    cosine_sim = cosine_similarity(csr_matrix(features_normalized), csr_matrix(features_normalized))

    return df, cosine_sim

@app.get('/recomendacion/{titulo}')
def get_recomendacion(titulo: str):
    df, cosine_sim = compute_similarity()
    df['title_normalized'] = df['title'].apply(normalizar_nombre)

    titulo_normalizado = normalizar_nombre(titulo)
    try:
        idx = df[df['title_normalized'] == titulo_normalizado].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Película no encontrada.")

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # 5 mejores recomendaciones
    movie_indices = [i[0] for i in sim_scores]

    return {"recomendaciones": df['title'].iloc[movie_indices].tolist()}

import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Iniciando en puerto: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
