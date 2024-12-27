# B351-Final-Project

This project uses FastAPI for backend services, PostgreSQL for the database, and Docker Compose for managing containerized environments. Follow these instructions to set up and run the project on your local environment.

![Alt text](images/image1.png)

![Screenshot](images/image2.png)

![Screenshot](images/image3.png)

## Install the dependencies

```bash 
pip install -r requirements.txt
```

## Install the Postgresql database

```bash 
brew install postgresql@15
brew services start postgresql@15
```
## Running the FAST-API


```bash 
uvicorn main:app
```

On your web browser, go to http://127.0.0.1:8000/docs and add the image for prediction at /predictor


## Running the video

```bash 
cd video_test
python3 video.py
```



## Using Docker-Compose and CI/CD

Install Docker-compose

```bash
brew install docker-compose
```
Build ther Docker-compose

```bash 
docker-compose up --build
```

Stop the docker-compose

```bash 
docker-compose down
```


## License

[MIT](https://choosealicense.com/licenses/mit/)


