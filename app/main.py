from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from .predict import make_prediction
from pydantic import BaseModel

class PredictionSchema(BaseModel):
    filename    : str
    content_type: str
    prediction  : int

app = FastAPI(title="Skin cancer prediction")

allowed_formats = ['jpg','jpeg','png']


@app.get('/')
def home():
    return 'Skin-cancer-diagnosis with vgg19.  To use the api send the post request to /predict'

@app.post("/predict",response_model=PredictionSchema)
def predict(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in allowed_formats
    if not extension:
        return "Invalid format, it must be a jpg, jpeg or png"
    image = file.file
    print(type(image))
    prediction = make_prediction(image)
    print(prediction)
    return PredictionSchema(
        filename    =file.filename,
        content_type=file.content_type,
        prediction  =prediction
    )
