import uvicorn
from typing import List
from fastapi import FastAPI, UploadFile, File, Request, Form
import shutil


from keras.models import load_model
import tensorflow as tf
import keras.utils as image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

#image upload_image
from fastapi.staticfiles import StaticFiles
from PIL import Image
import secrets



# web
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates



app = FastAPI()

app.mount('/static', StaticFiles(directory= "static"), name = "static")


templates = Jinja2Templates(directory="templates")

@app.get('/index_get', response_class = HTMLResponse)
def index_get (request : Request):
    context = {'request' : request}
    return templates.TemplateResponse("index.html", context)

@app.post('/uploadfile/image')
async def create_upload_file (file : UploadFile = File(...)):
    filepath = "static/images"
    filename = file.filename
    extension = filename.split(".")[1]
    if extension not in ["png", "jpg", "jpeg", "JPEG", "JPG", "PNG"]:
        return  {"status" : "error", "detail" : "file extension not allowed"}

    token_name = secrets.token_hex(12) + "."+extension
    generated_name = filepath + token_name
    file_content = await file.read()
    with open (generated_name, "wb") as file1:
        file1.write(file_content)
    img = Image.open(generated_name)
    img = img.resize(size = (224,224))
    img.save(generated_name)
    file1.close()

    loaded_m = load_model('VGG.h5')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = loaded_m.predict(x)
    result = decode_predictions(features)
    arr = []
    for i in result[0]:
        arr.append(i[2])
    arr = np.array(arr)
    a = np.argmax(arr)
    return {'class'  :f'{result[0][a][1]}'}
#
#


if __name__ == '__main__':
    uvicorn.run(app,host = '127.0.0.1',port= '8000')
#uvicorn main:app --reload
#pip freeze > requirements.txt
