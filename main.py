import os
import joblib
import pandas as pd
from fastapi import FastAPI
from supabase import create_client

app = FastAPI()

# Loading the 'binary' file you just uploaded
model = joblib.load("model.joblib")

# Supabase Connection
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY")
supabase = create_client(url, key)

@app.get("/")
def home():
    return {"status": "Backend is Running"}

@app.post("/predict")
async def predict(data: dict):
    # This takes the user input and feeds it to the model
    input_df = pd.DataFrame([{
        'KARMAŞIKLIK': data['complexity'],
        'BOYUT TİPİ': data['size_type'],
        'KAÇ TİP BOYA': data.get('paint_count', 1),
        'EN (CM)': data['en'],
        'BOY (CM)': data['boy'],
        'DERİNLİK (CM)': data['depth']
    }])
    
    prediction = model.predict(input_df)[0]
    
    # Save to SQL
    supabase.table("prediction_logs").insert({
        "part_no": data.get("part_no", "N/A"),
        "predicted_hours": float(prediction)
    }).execute()
    
    return {"prediction": round(prediction, 4)}
