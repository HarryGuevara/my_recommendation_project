from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200, "El código de estado debe ser 200"
    assert response.json() == {"message": "Bienvenido a la API de análisis de películas"}, "El mensaje de bienvenida no es el esperado"

def test_cantidad_filmaciones_mes():
    # Usa el nombre del mes en español
    response = client.get("/cantidad_filmaciones_mes/enero")
    assert response.status_code == 200, "El código de estado debe ser 200"
    
    json_response = response.json()
    assert "cantidad" in json_response, "La respuesta debe contener el campo 'cantidad'"
    assert isinstance(json_response["cantidad"], int), "La cantidad debe ser un entero"
    assert json_response["cantidad"] >= 0, "La cantidad no puede ser negativa"

def test_datos_unido():
    response = client.get("/datos_unido")
    assert response.status_code == 200, "El código de estado debe ser 200"
    
    json_response = response.json()
    assert isinstance(json_response, (list, dict)), "La respuesta debe ser una lista o un diccionario"
    
    # Si es una lista, verifica que el primer ítem sea un diccionario
    if isinstance(json_response, list) and len(json_response) > 0:
        assert isinstance(json_response[0], dict), "El primer elemento de la lista debe ser un diccionario"
        # Puedes agregar más verificaciones específicas para la estructura del diccionario
        assert "id_film" in json_response[0], "El primer elemento debe contener el campo 'id_film'"
        assert "cast_id" in json_response[0], "El primer elemento debe contener el campo 'cast_id'"

# Más pruebas pueden ser agregadas aquí para cubrir otros endpoints de la API
