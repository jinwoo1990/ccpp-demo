FROM python:3.7-stretch
WORKDIR /api/
ENV TZ=Asia/Seoul
ENV VIRTUAL_ENV=/api/.venv
ENV PATH=${VIRTUAL_ENV}/bin:$PATH
RUN python -m venv ${VIRTUAL_ENV}
COPY requirements.txt /api/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /api/logs
COPY . .
ENTRYPOINT ["python"]
CMD ["app.py"]
