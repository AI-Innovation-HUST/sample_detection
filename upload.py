import os
import cv2
import numpy as np
from milvus_helpers import connect_milvus, create_collection, insert_data
from pipeline import Pipeline

def add_images_from_folder(folder_path, id_object):
    
    collection_name = f"object_{id_object}"
    try:
        collection = create_collection(collection_name)
    except Exception as e:
        print(f"Failed to create collection {collection_name}: {e}")


    pipe = Pipeline(model_path="saved_models\\isnet-general-use.pth")
    
    ids = 0
    for img in os.listdir(folder_path):
        img_path = (os.path.join(folder_path, img))
        img = cv2.imread(img_path)
        embeddings = []
        embedding = pipe.extract_emb(img)
        embeddings.append(embedding)
        insert_data(collection, ids, embeddings, img_path)
        ids += 1
    

if __name__ == "__main__":
    connect_milvus()
    folder_path = "C:\\Users\\7400\\Downloads\\pipeline\\DIS\\AI_test\\test"
    id_object = "1"
    add_images_from_folder(folder_path, id_object)
