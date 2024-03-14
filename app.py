#import libraries we need
import uvicorn # provide an asynchrone interface between python web app and web servers, necessary with fastapi
from fastapi import FastAPI # help building API in an easy way and good performance
from pydantic import BaseModel # basemodel is a class used to define data models
# from typing import Literal, Union # typing provides a set of classes for working with types and type hints abd specify expected types of variables for example
from joblib import load
from functions import calculate_all_indicators_optimised
import yfinance as yf

# description
description = """
Estimation of bitcoin price for tomorrow based on last datas, with a specific machine learning models (linear regression, elasticnet or xgboost).
the input is a dictionary with the following items :
Model : string
"""

# tags to identify different endpoints
tags_metadata = [{"name": "ML-Model-Prediction","description": "Estimation of bitcoin price with a specific machine learning model"}]

app = FastAPI(
    title="Bitcoin price API",
    description=description,
    version="0.1",
    contact={
        "name": "sndd",
        "url": "https://github.com/sdupland",
    },
    openapi_tags=tags_metadata)

# define for each feature the type of data (integer, boulean, float), the default value, and if necessary a list of possible values
class Features(BaseModel):
    Model : str

# decorator in FastAPI that associates the function message() with the specified route between coma ("/") and the HTTP GET method
@app.get("/")
# The message() function is defined as an asynchronous function
# This allows asynchronous operations to be performed within the function.
# When someone accesses the root URL ("/") of the application using a web browser or any HTTP client, they will receive the text "Estimation of rental price with xgboost machine learning model" as the response.
async def message() :
    texte = "Estimation of bitcoin price with a specific machine learning model"
    return texte


# uses the @app.post decorator to associate the function predict() with the HTTP POST method at the "/predict" endpoint
@app.post("/predict", tags=["ML-Model-Prediction"])
# the predict function is defined as an asynchronous function
# It takes a parameter Features
# it expects a request body containing data with a structure defined by the Features model.
async def predict(Features: Features):
    """
    Calculate the Bitcoin for tomorrow
    
    params : input needs a dictionary of values for the following items :
    Model : string

    return : the function returns a float number which is the predicted bitcoin price for the model given in input
    """
    btc_ticker = yf.Ticker("BTC-EUR")
    btc_actual = btc_ticker.history(period="200d", actions=False)
    btc_actual = btc_actual.tz_localize(None)
    btc_actual = calculate_all_indicators_optimised(btc_actual)
    btc_actual = btc_actual.sort_index(ascending=False)
    input_data = btc_actual.iloc[0,:]
    input_data = input_data.fillna(0)
    input_data = input_data.array
    input_data = input_data.reshape(1,-1)
    scaler = load("scaler.bin")
    input_data = scaler.transform(input_data)
    
    Model = Features.Model
    if Model =="Linear regression" :
        path = "model_rfe_lr.joblib"
        model = load(path)
    elif Model == "Elastic net" :
        path = "model_elasticnet.joblib"
        model = load(path)
    else :
        path = "model_rfe_xgb.joblib"
        model = load(path)
        
    futur_price = model.predict(input_data)
    last_price = btc_actual.iloc[0,3]
    response = {"Model" : Model, "futur_price" : round(futur_price.tolist()[0],2), "last_price" : round(last_price,2)}
    return response

if __name__ == "__main__": # checks if the script is being executed as the main program
    #  If the script is being run as the main program (i.e., not imported as a module), this line is executed. It uses the uvicorn.run() function to run a FastAPI application (app) using the Uvicorn ASGI server.
    uvicorn.run(app, host="0.0.0.0", port=4000)
