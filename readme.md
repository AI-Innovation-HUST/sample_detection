Install Milvus

`cd docker`
`docker compose up -d`


Install Milvus Interface (Antu)

`docker run -p 8000:3000  -e MILVUS_URL={your machine IP}:19530 zilliz/attu:latest`

for example, my IP is: `192.168.1.10`

Run `app.py`

IP Web Local `http://127.0.0.1:5000/`

