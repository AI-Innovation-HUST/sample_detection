import os
import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from milvus_helpers import connect_milvus, create_collection, insert_data, search_data, list_collections, drop_collection
from pipeline import Pipeline

app = Flask(__name__)

# Kết nối tới Milvus
connect_milvus()

# Tạo pipeline trích xuất đặc trưng
pipe = Pipeline(model_path="saved_models/isnet-general-use.pth")

# Trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Thêm hình ảnh từ thư mục
@app.route('/add_images', methods=['GET', 'POST'])
def add_images():
    message = ""
    if request.method == 'POST':
        folder_path = request.form['folder_path']
        id_object = int(request.form['id_object'])
        if os.path.exists(folder_path):
            add_images_from_folder(folder_path, id_object)
            message = f"Added images from {folder_path} to collection object_{id_object}"
        else:
            message = "Folder path does not exist"
    return render_template('add_images.html', message=message)

# Tìm kiếm hình ảnh
@app.route('/search', methods=['GET', 'POST'])
def search():
    results = []
    if request.method == 'POST':
        image_file = request.files['image']
        id_object = int(request.form['id_object'])
        if image_file:
            image = Image.open(image_file.stream)
            results = query_image(image, id_object)
    return render_template('search.html', results=results)

# Xóa collection
@app.route('/drop_collection', methods=['GET', 'POST'])
def drop_collection_route():
    message = ""
    if request.method == 'POST':
        collection_name = request.form['collection_name']
        drop_collection(collection_name)
        message = f"Dropped collection {collection_name}"
    return render_template('drop_collection.html', message=message)

# Liệt kê các collection
@app.route('/list_collections', methods=['GET'])
def list_collections_route():
    collections = list_collections()
    return render_template('list_collections.html', collections=collections)

def add_images_from_folder(folder_path, id_object):
    collection_name = f"object_{id_object}"
    try:
        collection = create_collection(collection_name)
    except Exception as e:
        print(f"Failed to create collection {collection_name}: {e}")
    ids = 0
    for img in os.listdir(folder_path):
        img_path = (os.path.join(folder_path, img))
        img = cv2.imread(img_path)
        embeddings = []
        embedding = pipe.extract_emb(img)
        embeddings.append(embedding)
        insert_data(collection, ids, embeddings, img_path)
        ids += 1

def query_image(image, id_object, top_k=1):
    embedding = pipe.extract_emb(np.array(image))
    collection_name = f"object_{id_object}"
    collection = create_collection(collection_name)
    collection.load()  # Ensure collection is loaded before searching
    results = search_data(collection, embedding, top_k)
    matched_files = [result.entity.get("file_name") for result in results[0]]
    return matched_files


@app.teardown_appcontext
def cleanup(exception=None):
    torch.cuda.empty_cache()

if __name__ == "__main__":
    app.run(debug=True)

