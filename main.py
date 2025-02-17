from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import unicodedata
from fuzzywuzzy import process

app = FastAPI()

# Cargar solo las películas más populares (30,000 más populares)
movies_df = pd.read_csv('data/movies_dataset.csv')
movies_df = movies_df.sort_values(by='popularity', ascending=False).head(30000)

# Cargar actores y directores más frecuentes
cast_df = pd.read_csv('data/cast.csv')
crew_df = pd.read_csv('data/crew.csv')

# Filtrar actores que aparecen en al menos 5 películas
actor_counts = cast_df['name_actor'].value_counts()
actores_frecuentes = actor_counts[actor_counts >= 5].index
cast_df = cast_df[cast_df['name_actor'].isin(actores_frecuentes)]

# Filtrar directores que dirigen al menos 3 películas
director_counts = crew_df[crew_df['job_crew'] == 'Director']['name_job'].value_counts()
directores_frecuentes = director_counts[director_counts >= 3].index
crew_df = crew_df[(crew_df['name_job'].isin(directores_frecuentes)) & (crew_df['job_crew'] == 'Director')]

# Cargar solo las columnas necesarias
movies_df = movies_df[['title', 'vote_average', 'popularity', 'release_year', 'revenue', 'budget', 'release_date']]
cast_df = cast_df[['movie_id', 'name_actor']]
crew_df = crew_df[['movie_id', 'name_job', 'job_crew']]

# Normalizar nombres
def normalizar_nombre(nombre):
    nombre = unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('ascii')
    return nombre.lower()

movies_df['title_normalized'] = movies_df['title'].apply(normalizar_nombre)
cast_df['name_actor_normalized'] = cast_df['name_actor'].apply(normalizar_nombre)
crew_df['name_job_normalized'] = crew_df['name_job'].apply(normalizar_nombre)

# Manejo de valores NaN
movies_df['vote_average'].fillna(movies_df['vote_average'].mean(), inplace=True)
movies_df['popularity'].fillna(movies_df['popularity'].mean(), inplace=True)
movies_df['release_year'].fillna(movies_df['release_year'].mode()[0], inplace=True)

# Calcular 'return' y manejar divisiones por cero
movies_df['return'] = movies_df.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] != 0 else 0, axis=1)

# Seleccionar características relevantes
features = movies_df[['vote_average', 'popularity', 'release_year', 'return']]

# Normalizar las características
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Calcular la matriz de similitud del coseno
cosine_sim = cosine_similarity(features_normalized, features_normalized)

# Función de recomendación
def recomendacion(titulo: str, cosine_sim=cosine_sim, movies_df=movies_df):
    titulo_normalizado = normalizar_nombre(titulo)
    idx = movies_df[movies_df['title_normalized'] == titulo_normalizado].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

# Endpoints (sin cambios)
@app.get('/')
def read_root():
    return {"message": "Bienvenido a la API de recomendación de películas"}

# ... (resto de los endpoints sin cambios)
