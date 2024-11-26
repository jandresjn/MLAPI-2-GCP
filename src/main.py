from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import pandas as pd
import numpy as np
import joblib

from .configuraciones import config

app = FastAPI()

ruta_actual = os.getcwd()

def prediccion_o_inferencia(pipeline_de_test, datos_de_test):
    #Dropeamos
    datos_de_test.drop('Id', axis=1, inplace=True)
    # Cast MSSubClass as object
    datos_de_test['MSSubClass'] = datos_de_test['MSSubClass'].astype('O')
    datos_de_test = datos_de_test[config.FEATURES] #ESTA ES LA UNICA LINEA DIFERENTE DEL JUPYTER

    new_vars_with_na = [
        var for var in config.FEATURES
        if var not in config.CATEGORICAL_VARS_WITH_NA_FREQUENT +
        config.CATEGORICAL_VARS_WITH_NA_MISSING +
        config.NUMERICAL_VARS_WITH_NA
        and datos_de_test[var].isnull().sum() > 0]
    
    datos_de_test.dropna(subset=new_vars_with_na, inplace=True)

    #AQUI ESTOY HACIENDO LA INGENIERIA DE DATOS DE PIPELINE Y LA INFERENCIA DEL MODELO DE ML
    predicciones = pipeline_de_test.predict(datos_de_test)
    #ESTOY DESESCALANDO
    predicciones_sin_escalar = np.exp(predicciones)
    return predicciones, predicciones_sin_escalar, datos_de_test

@app.get("/") #Endpoint Raiz
def print_get():
    return {"mensaje": "Testando API"}

@app.get("/ruta-actual")
def fun_ruta_actual():
    return {f"mensaje: {ruta_actual}"}

# @app.get()


@app.post("/publicar_mensaje")
async def publicar_mensaje(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Guarda el archivo en una ubicación temporal
    file_location = ruta_actual + "/" + file.filename
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    # Lee el archivo usando pandas
    df_de_los_datos_subidos = pd.read_csv(file_location)

    pipeline_de_produccion = joblib.load("./src/precio_casas_pipeline.joblib")

    predicciones, predicciones_sin_escalar, datos_test_procesados = (
        prediccion_o_inferencia(pipeline_de_produccion, df_de_los_datos_subidos)
    )

    df_concatenado = pd.concat(
        [
            datos_test_procesados,
            pd.Series(predicciones),
            pd.Series(predicciones_sin_escalar),
        ],
        axis=1,
    )

    output_file = ruta_actual + "/salida_datos_y_predicciones.csv"
    df_concatenado.to_csv(output_file, index=False)

    return FileResponse(
        output_file,
        media_type="application/octet-stream",
        filename="salida_datos_y_predicciones.csv",
    )
