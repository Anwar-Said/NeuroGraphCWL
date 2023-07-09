FROM python:3.9.15-slim-buster
WORKDIR /app
RUN pip3 install nilearn nibabel NeuroGraph numpy torch torch_geometric pandas
COPY . .
COPY prepare.py .