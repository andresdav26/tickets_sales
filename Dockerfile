FROM python:3.9

WORKDIR /Users/USUARIO/Documents/challenger_chiper/src
ENV PYTHONPATH=$PYTHONPATH:/Users/USUARIO/Documents/challenger_chiper/

COPY requeriments.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r requeriments.txt --no-cache-dir

COPY api/. .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-config", "./src/api/logging.conf"]


