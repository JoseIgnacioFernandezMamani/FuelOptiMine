#!/bin/bash

# Iniciar todos los servicios automáticamente
set -e

echo "Iniciando ClickHouse..."
sudo systemctl start clickhouse-server

echo "Iniciando Django (backend)..."
cd ../backend && python manage.py runserver 0.0.0.0:8000 &

echo "Iniciando Streamlit (frontend)..."
cd ../frontend && streamlit run app.py --server.port 8501 &

echo "Servicios activos:"
echo "- ClickHouse: localhost:9000"
echo "- Django:     localhost:8000"
echo "- Streamlit:  localhost:8501"