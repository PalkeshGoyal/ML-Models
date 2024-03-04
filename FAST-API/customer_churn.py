from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel,Field
import pandas as pd
import joblib 
from fastapi.responses import JSONResponse

gb_model = joblib.load("./gb_model.pkl")

class Gender(str, Enum):
    Male = "Male"
    Female = "Female"


class Geography(str, Enum):
    France = "France"
    Germany = "Germany"
    Spain = "Spain"



class Customer(BaseModel):
    CreditScore: int = Field(gt=0)
    Geography: Geography
    Gender: Gender
    Age: int = Field(gt=0)
    Tenure : int  = Field(gt=0,le=10)
    Balance : float = Field(ge=0)
    NumOfProducts : int  = Field(gt=0,le=4)
    HasCrCard : int = Field(ge=0,le=1)
    IsActiveMember : int  = Field(ge=0 , le=1)
    EstimatedSalary : float   = Field(ge=0)



app = FastAPI()


@app.post("/predict/")
async def predict(customer: Customer):
    Geography_dict = {"France" : 0.161612, "Germany":0.324701,"Spain":0.166801}
    gender_dict = {"Female": 0.250715 , "Male" : 0.164803}
    customer.Geography = Geography_dict[customer.Geography]
    customer.Gender = gender_dict[customer.Gender]
    res = predict(customer)
    res = res.tolist()
    return JSONResponse(content={"Exited":res[0]},media_type="application/json")

def predict(df):
    df = [dict(df)]
    df = pd.DataFrame.from_dict(df)
    return gb_model.predict(df)