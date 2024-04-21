from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
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

from core.config import settings
from db.session import engine
from db.base_class import Base
from db.session import SessionLocal
from sqlalchemy.orm import Session
from db.base_class import ImageData, LicensePlateData
import cv2
import logging
import numpy as np

from easyocr import Reader
def create_image_data(db: Session, filename: str, image_size: int):
    db_image = ImageData(filename=filename, image_size=image_size)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image

def create_tables():         
	Base.metadata.create_all(bind=engine)

def create_license_plate_data(db: Session, image_name: str, plate_text: str) -> LicensePlateData:
    license_plate_data = LicensePlateData(image_name=image_name, plate_text=plate_text)
    db.add(license_plate_data)
    db.commit()
    db.refresh(license_plate_data)
    return license_plate_data

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
create_tables()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
 
#predictor = HelmetPredictor('tracker/model_weights.pth')
predictor = HelmetPredictor("model/runs/detect/yolov8n_custom3/weights/best.pt", model_type="yolov8")
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




    
@app.post("/predictor")
async def predict_file(file: UploadFile, db: Session = Depends(get_db)):
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
        image_draw, license_plate_text = predictor.draw_boxes(image)

        
        logging.info(f"License plate: {license_plate_text}")
        numpy_image = np.array(image_draw)
        cv2.imshow("image", image_draw)
        cv2.imwrite(image_filename, numpy_image)
        def iter_file():
            with open(image_filename, mode="rb") as file_like:
                yield from file_like

        #license_plate_text = "AJJJING"
        create_license_plate_data(db, image_filename, license_plate_text)

        return StreamingResponse(iter_file(), media_type="image/png")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))