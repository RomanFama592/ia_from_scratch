FROM python:3.8

WORKDIR /app

EXPOSE 8000

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./server_nn.py" ]