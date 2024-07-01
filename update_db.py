import torch
import clip
from PIL import Image
import os
from qdrant_client import models, QdrantClient
import yaml

#Load config file
with open("config/config.yml","r") as file:
    config=yaml.safe_load(file)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(config["model_name"], device=device)
qdrant = QdrantClient(config["server_qdrant"],port=config["port"])


# Create logo database
qdrant.recreate_collection(
    collection_name=config["collection_name"],
    vectors_config=models.VectorParams(
        size=512, # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

'''upload emb to db'''
image_list_save_faiss=[]
db=[]
for roots,dirs,files in os.walk(config["folder_db"]):
    if len(dirs)==0:
        for file in files:
            step=dict()
            image_path=os.path.join(roots,file)
            label=os.path.basename(roots)
            img=Image.open(image_path)
            input=preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(input)
            image_features=torch.squeeze(image_features,0)
            output=image_features.detach().cpu().numpy()
            step["image_name"]=image_path
            step["label"]=label
            step["image_emb"]=output.tolist()
            # print(output.shape)
            image_list_save_faiss.append(file)
            db.append(step)
# Let's vectorize descriptions and upload to qdrant
try:
    qdrant.upload_records(
        collection_name=config["collection_name"],
        records=[
            models.Record(
                id=idx,
                vector=doc["image_emb"],
                payload=doc
            ) for idx, doc in enumerate(db)
        ]
    )
    print("End task")
except:
    print("Done task")