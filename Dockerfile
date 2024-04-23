# Base image with Python and other dependencies
FROM python:3.9

# Install required system packages, including those for OpenCV
RUN apt-get update && \
    apt-get install -y gcc python3-dev build-essential libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

RUN pip install opencv-python
# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the FastAPI app code into the working directory
COPY . .

# Expose the port on which FastAPI runs (default: 8000)
EXPOSE 8000

# Command to run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
