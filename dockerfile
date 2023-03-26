FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["python", "/code/run.py", "--address", "0.0.0.0", "--port", "7860", "--allow-websocket-origin", "127.0.0.1:5000"]
