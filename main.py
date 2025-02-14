from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from memory_profiler import profile

app = FastAPI()

@profile
def cargar_datos():
    global movies_df, cast_df, crew_df
    movies_df = pd.read_csv('data/movies_dataset.csv')
    cast_df = pd.read_csv('data/cast.csv')
    crew_df = pd.read_csv('data/crew.csv')

cargar_datos()


# Manejo de valores NaN
# Rellenar valores NaN en vote_average y popularity con la media
movies_df['vote_average'].fillna(movies_df['vote_average'].mean(), inplace=True)
movies_df['popularity'].fillna(movies_df['popularity'].mean(), inplace=True)

# Rellenar valores NaN en release_year con la moda (año más común)
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
@profile
def recomendacion(titulo: str, cosine_sim=cosine_sim, movies_df=movies_df):
    idx = movies_df[movies_df['title'].str.lower() == titulo.lower()].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

# Endpoint de bienvenida
@profile
@app.get('/')
def read_root():
    return {"message": "Bienvenido a la API de recomendación de películas"}

# Endpoint 1: cantidad_filmaciones_mes
@profile
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
    
    return {"mensaje": f"{cantidad} películas fueron estrenadas en el mes de {mes.capitalize()}"}

# Endpoint 2: cantidad_filmaciones_dia
@profile
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
    
    return {"mensaje": f"{cantidad} películas fueron estrenadas en los días {dia.capitalize()}"}

# Endpoint 3: score_titulo
@profile
@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    titulo_pelicula = pelicula['title'].values[0]
    año_estreno = pelicula['release_year'].values[0]
    score = pelicula['vote_average'].values[0]
    
    return {"mensaje": f"La película {titulo_pelicula} fue estrenada en el año {año_estreno} con un score/popularidad de {score}"}

# Endpoint 4: votos_titulo
@profile
@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    if pelicula['vote_count'].values[0] < 2000:
        return {"mensaje": f"La película {titulo} no cumple con la condición de tener al menos 2000 valoraciones."}
    
    titulo_pelicula = pelicula['title'].values[0]
    año_estreno = pelicula['release_year'].values[0]
    votos = pelicula['vote_count'].values[0]
    promedio = pelicula['vote_average'].values[0]
    
    return {"mensaje": f"La película {titulo_pelicula} fue estrenada en el año {año_estreno}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio}"}

# Endpoint 5: get_actor
@profile
@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    # Filtrar películas del actor
    actor_peliculas = cast_df[cast_df['name_actor'].str.lower() == nombre_actor.lower()]
    
    if actor_peliculas.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado.")
    
    # Obtener IDs de las películas
    peliculas_ids = actor_peliculas['movie_id'].unique()
    
    # Filtrar películas del actor en el dataset de películas
    peliculas_actor = movies_df[movies_df['movie_id'].isin(peliculas_ids)]
    
    if peliculas_actor.empty:
        raise HTTPException(status_code=404, detail="No se encontraron películas para este actor.")
    
    # Calcular métricas
    cantidad_peliculas = peliculas_actor.shape[0]
    retorno_total = peliculas_actor['return'].sum()
    promedio_retorno = retorno_total / cantidad_peliculas if cantidad_peliculas > 0 else 0
    
    return {"mensaje": f"El actor {nombre_actor} ha participado de {cantidad_peliculas} filmaciones, el mismo ha conseguido un retorno de {retorno_total} con un promedio de {promedio_retorno} por filmación"}


# Endpoint 6: get_director
@profile
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    # Filtrar películas del director
    director_peliculas = crew_df[(crew_df['name_job'].str.lower() == nombre_director.lower()) & (crew_df['job_crew'] == 'Director')]
    
    if director_peliculas.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado.")
    
    # Obtener IDs de las películas
    peliculas_ids = director_peliculas['movie_id'].unique()
    
    # Filtrar películas del director en el dataset de películas
    peliculas_director = movies_df[movies_df['movie_id'].isin(peliculas_ids)]
    
    if peliculas_director.empty:
        raise HTTPException(status_code=404, detail="No se encontraron películas para este director.")
    
    # Calcular métricas
    retorno_total = peliculas_director['return'].sum()
    detalles_peliculas = peliculas_director[['title', 'release_date', 'return', 'budget', 'revenue']].to_dict('records')
    
    return {
        "mensaje": f"El director {nombre_director} ha conseguido un retorno total de {retorno_total}",
        "peliculas": detalles_peliculas
    }
# Endpoint de recomendación
@profile
@app.get('/recomendacion/{titulo}')
def get_recomendacion(titulo: str):
    try:
        recomendaciones = recomendacion(titulo)
        return {"recomendaciones": recomendaciones}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error: {str(e)}")
