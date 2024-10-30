from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar el modelo desde el archivo .pkl
loaded_model = joblib.load("coffee_price_model.pkl")

# Crear la instancia de FastAPI
app = FastAPI()

# Definir el modelo de datos para la solicitud
class PredictionRequest(BaseModel):
    coffee_acreage: float
    coffee_improved_acreage: float
    coffee_improved_cost: float
    coffee_acreage_fertilizer: float
    coffee_fertilizer_cost: float
    coffee_chemical_acreage: float
    coffee_chemical_cost: float
    coffee_machinery_acreage: float
    coffee_machinery_cost: float
    coffee_harvested: float
    coffee_sold_price: float
    coffee_harvest_loss: float

@app.post("/predict")
def predict(request: PredictionRequest):
    # Extraer los datos de la solicitud y convertirlos a un DataFrame
    input_data = pd.DataFrame([[
        request.coffee_acreage,
        request.coffee_improved_acreage,
        request.coffee_improved_cost,
        request.coffee_acreage_fertilizer,
        request.coffee_fertilizer_cost,
        request.coffee_chemical_acreage,
        request.coffee_chemical_cost,
        request.coffee_machinery_acreage,
        request.coffee_machinery_cost,
        request.coffee_harvested,
        request.coffee_sold_price,
        request.coffee_harvest_loss
    ]], columns=[
        'coffee_acreage', 'coffee_improved_acreage', 'coffee_improved_cost',
        'coffee_acreage_fertilizer', 'coffee_fertilizer_cost', 'coffee_chemical_acreage',
        'coffee_chemical_cost', 'coffee_machinery_acreage', 'coffee_machinery_cost',
        'coffee_harvested', 'coffee_sold_price', 'coffee_harvest_loss'
    ])

    # Realizar la predicción
    try:
        prediction = loaded_model.predict(input_data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"Predicción de Cafe": prediction}

# Endpoint de prueba para la raíz
@app.get("/")
def read_root():
    return {"message": "Predicción de Cafe lista"}

# Si ejecutas el archivo directamente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)