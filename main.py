from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import unicodedata

app = FastAPI()

# 1️⃣ Cargar solo el archivo optimizado
movies_df = pd.read_csv(
    'data/top_20000_movies.csv',
    dtype={
        'popularity': 'float32',
        'vote_average': 'float32',
        'release_year': 'int32',
        'revenue': 'float32',
        'budget': 'float32',
        'movie_id': 'int32'
    }
)

# 2️⃣ Normalizar nombres
def normalizar_nombre(nombre):
    nombre = unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('ascii')
    return nombre.lower()

movies_df['title_normalized'] = movies_df['title'].apply(normalizar_nombre)

# 3️⃣ Manejo de valores NaN
movies_df['vote_average'].fillna(movies_df['vote_average'].mean(), inplace=True)
movies_df['popularity'].fillna(movies_df['popularity'].mean(), inplace=True)
movies_df['release_year'].fillna(movies_df['release_year'].mode()[0], inplace=True)

# 4️⃣ Calcular retorno
movies_df['return'] = np.where(
    movies_df['budget'] != 0,
    movies_df['revenue'] / movies_df['budget'],
    0
)

# 5️⃣ Crear la matriz de similitud (recomendaciones)
features = movies_df[['vote_average', 'popularity', 'release_year', 'return']]
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)
cosine_sim = cosine_similarity(csr_matrix(features_normalized), csr_matrix(features_normalized))

# 6️⃣ Función de recomendación
def recomendacion(titulo: str):
    titulo_normalizado = normalizar_nombre(titulo)
    try:
        idx = movies_df[movies_df['title_normalized'] == titulo_normalizado].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Las 5 mejores recomendaciones
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

# ✅ ENDPOINTS
@app.get('/')
def read_root():
    return {"message": "Bienvenido a la API de recomendación de películas"}

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    mes = mes.lower()
    if mes not in meses:
        raise HTTPException(status_code=400, detail="Mes no válido.")
    
    peliculas_mes = movies_df[movies_df['release_date'].notna()]
    peliculas_mes['release_month'] = pd.to_datetime(peliculas_mes['release_date']).dt.month
    cantidad = peliculas_mes[peliculas_mes['release_month'] == meses[mes]].shape[0]
    
    return {"mensaje": f"{cantidad} películas fueron estrenadas en {mes.capitalize()}"}

@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    dias = {
        'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3,
        'viernes': 4, 'sábado': 5, 'domingo': 6
    }
    dia = dia.lower()
    if dia not in dias:
        raise HTTPException(status_code=400, detail="Día no válido.")
    
    peliculas_dia = movies_df[movies_df['release_date'].notna()]
    peliculas_dia['release_day'] = pd.to_datetime(peliculas_dia['release_date']).dt.weekday
    cantidad = peliculas_dia[peliculas_dia['release_day'] == dias[dia]].shape[0]
    
    return {"mensaje": f"{cantidad} películas fueron estrenadas en {dia.capitalize()}"}

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    return {
        "mensaje": f"La película {pelicula['title'].values[0]} fue estrenada en {pelicula['release_year'].values[0]} con un score de {pelicula['vote_average'].values[0]}"
    }

@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    if 'vote_count' not in pelicula.columns or pelicula['vote_count'].values[0] < 2000:
        return {"mensaje": f"La película {titulo} no tiene al menos 2000 valoraciones."}
    
    return {
        "mensaje": f"La película {pelicula['title'].values[0]} fue estrenada en {pelicula['release_year'].values[0]} con {pelicula['vote_count'].values[0]} valoraciones y un promedio de {pelicula['vote_average'].values[0]}"
    }

@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    # Suponiendo que el archivo top_20000_movies.csv incluye una columna 'actors' con nombres de actores
    actor_peliculas = movies_df[movies_df['actors'].str.contains(nombre_actor, case=False, na=False)]
    
    if actor_peliculas.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado.")
    
    return {
        "mensaje": f"El actor {nombre_actor} ha participado en las siguientes películas:",
        "peliculas": actor_peliculas[['title', 'release_date']].to_dict('records')
    }

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    # Suponiendo que el archivo top_20000_movies.csv incluye una columna 'directors' con nombres de directores
    director_peliculas = movies_df[movies_df['directors'].str.contains(nombre_director, case=False, na=False)]
    
    if director_peliculas.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado.")
    
    retorno_total = director_peliculas['return'].sum()
    
    return {
        "mensaje": f"El director {nombre_director} ha conseguido un retorno total de {retorno_total}",
        "peliculas": director_peliculas[['title', 'release_date', 'return']].to_dict('records')
    }

@app.get('/recomendacion/{titulo}')
def get_recomendacion(titulo: str):
    try:
        recomendaciones = recomendacion(titulo)
        return {"recomendaciones": recomendaciones}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error: {str(e)}")

import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Iniciando en puerto: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
