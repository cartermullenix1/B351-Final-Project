from fastapi import FastAPI, File, UploadFile
from typing_extensions import Annotated
from typing import Union
app = FastAPI()
 
db = [
    {"student_id" : "20241",
     "name": "Tri"},
     {"student_id" : "20242",
     "name": "Carter"},
     {"student_id" : "20243",
     "name": "Kush"},
 
]
 
@app.get("/")
async def root():
    return "Carter redneck"
 
@app.get("/student_db/{student_id}")
async def get_student(student_id:str):
    for record in db:
        if record["student_id"] == student_id:
            return record
        
@app.post("/student_db")
async def create_student(student_id:str, name):
    record = {
        "student_id": student_id,
        "name": name
    }
    db.append(record)
 
@app.put("/student_db/{student_id}")
async def update_student(student_id:str, name):
    for record in db:
        if record["student_id"] == student_id:
            record["name"] = name
 
@app.delete("/student_db/{student_id}")
async def update_student(student_id:str, name):
    for record in db:
        if record["student_id"] == student_id:
            db.remove(record)
 
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