FROM python:3.9.15-slim-buster
WORKDIR /app
RUN pip3 install nilearn nibabel numpy torch torch_geometric NeuroGraph pandas
COPY . .
COPY parcellation.py .
COPY remove_drifts.py .
COPY remove_h_motions.py .
COPY corr.py .
COPY prepare.py .
COPY utils.py .
COPY gcn.py .
COPY test.py .
