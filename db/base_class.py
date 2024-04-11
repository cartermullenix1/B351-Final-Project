from typing import Any
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import as_declarative
from sqlalchemy import Column, Integer, String


@as_declarative()
class Base:
    id: Any
    __name__: str

    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

class ImageData(Base):
    __tablename__ = "image_data"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    image_size = Column(Integer)

class LicensePlateData(Base):
    __tablename__ = "license_plate_data"

    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String, index=True)
    plate_text = Column(String)