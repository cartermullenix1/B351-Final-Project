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


 
predictor = HelmetPredictor('tracker/model_weights.pth')
storage_client = storage.Client.from_service_account_json('path/to/your/service-account-file.json')
bucket_name = 'your-bucket-name'


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