FROM python:3.9

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH = "${VIRTUAL_ENV}/bin:$PATH"

ENV TZ=America/Bogota

RUN python -m pip install --upgrade pip

COPY requeriments.txt .
RUN pip install -r requeriments.txt --no-cache-dir

RUN useradd --create-home adguerrero
USER adguerrero

WORKDIR /Users/USUARIO/Documents/challenger_chiper/
ENV PYTHONPATH=$PYTHONPATH:/Users/USUARIO/Documents/challenger_chiper/
COPY api/. .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-config", "./src/api/logging.conf"]


