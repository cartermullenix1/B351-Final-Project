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

    license_plates = get_license_plate_data(db)
    print("Record of License Plates")        
    for plate in license_plates:
        print(f"ID: {plate.id}, Image Name: {plate.image_name}, Plate Text: {plate.plate_text}")

    return license_plate_data

def get_license_plate_data(db: Session):
    return db.query(LicensePlateData).all()  # Retrieve all records


app = FastAPI()
# app.add_middleware(
#     CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
create_tables()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
 
# FasterRCNN model
#predictor = HelmetPredictor('tracker/model_weights.pth')
        
# YOLOv8 model
predictor = HelmetPredictor("model/runs/detect/yolov8n_custom4/weights/best.pt", model_type="yolov8")



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



# This is the "POST" use for prediction that we need to insert an image and the model 
# will predict the outcome

# Load the model predictor and try to predict the outcome
@app.post("/predictor")
async def predict_file(file: UploadFile, db: Session = Depends(get_db)):
    if not file:
        raise HTTPException(status_code=400, detail="No upload file sent")

    try:
        # This is tru
        image_data = await file.read()
        image_filename = "processed"+file.filename
        if not image_data:
            logging.error("No data read from file")
            raise HTTPException(status_code=400, detail="No data read from file")

        image_stream = BytesIO(image_data)
        image = Image.open(image_stream)
        # License text is trying to predict the image and also the license plate text
        image_draw, license_plate_text = predictor.draw_boxes(image)

        
        logging.info(f"License plate: {license_plate_text}")
        numpy_image = np.array(image_draw)
        #cv2.imshow("image", image_draw)
        cv2.imwrite(image_filename, numpy_image)
        def iter_file():
            with open(image_filename, mode="rb") as file_like:
                yield from file_like
        # This will try to send the image file name and license plate text to the database
        #license_plate_text = "AJJJING"
        create_license_plate_data(db, image_filename, license_plate_text)

        return StreamingResponse(iter_file(), media_type="image/png")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))