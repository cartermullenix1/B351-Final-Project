from fastapi import FastAPI, File, UploadFile
from typing_extensions import Annotated
from typing import Union
from starlette.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from predictor import HelmetPredictor
from PIL import Image
import io
from fastapi.responses import StreamingResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


 
predictor = HelmetPredictor()

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    if not file:
        return {"message": "No file sent"}
    else:
        return {"file_size": len(file)}
 
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    if not file:
        return {"message": "No upload file sent"}
    else:
        return {"filename": file.filename}

@app.post("/predictor")
async def predict_file(file: UploadFile):
    # if not file:
    #     return {"message": "No upload file sent"}
    # else:
    #     return {"filename": file.filename}
    
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Predict and draw boxes on the image
    image_with_boxes = predictor.predict_and_draw(image)

    # Convert PIL image to byte stream in PNG format
    img_byte_arr = io.BytesIO()
    image_with_boxes.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Return the image in the response
    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")