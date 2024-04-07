from fastapi import FastAPI, File, UploadFile, HTTPException
from typing_extensions import Annotated
from typing import Union
from starlette.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from predictor import HelmetPredictor
from PIL import Image
import io
from fastapi.responses import StreamingResponse
import logging
from io import BytesIO


app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


 
predictor = HelmetPredictor('tracker/model_weights.pth')

#predictor = 
#storage_client = storage.Client.from_service_account_json('path/to/your/service-account-file.json')
#bucket_name = 'your-bucket-name'


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


def upload_image_to_gcs(image_byte_arr, filename):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_string(image_byte_arr, content_type='image/png')


    
@app.post("/predictor")
async def predict_file(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No upload file sent")

    try:
        image_data = await file.read()
        image_filename = "processed"+file.filename
        if not image_data:
            logging.error("No data read from file")
            raise HTTPException(status_code=400, detail="No data read from file")

        image_stream = BytesIO(image_data)
        image = Image.open(image_stream)
        image_draw = predictor.predict_and_draw(image_stream)
        image_draw.save(image_filename)
        def iter_file():
            with open(image_filename, mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(iter_file(), media_type="image/png")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))