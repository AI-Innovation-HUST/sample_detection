
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, exceptions



def search_data(collection, embedding, top_k):
    collection.load()  # Ensure collection is loaded before searching
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([embedding], "embedding", search_params, limit=top_k, output_fields=["file_name"])
    return results



def connect_milvus():
    connections.connect("default", host="localhost", port="19530")

def create_collection(collection_name):
    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} already exists")
        collection = Collection(collection_name)
        collection.load()
        return collection 
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255)
    ]
    schema = CollectionSchema(fields, "Object Embeddings")
    collection = Collection(collection_name, schema)
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
    collection.create_index("embedding", index_params)
    collection.load()
    return collection

def insert_data(collection, ids, embeddings, file_names):
    id = []
    path = []
    id.append(ids)
    path.append(file_names)
    data = [id, embeddings, path]
    collection.insert(data)
    collection.load()  # Ensure collection is loaded after insertion

def list_collections():
    return utility.list_collections()

def drop_collection(collection_name):
    try:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.drop()
            return f"Collection {collection_name} dropped successfully"
        else:
            return f"Collection {collection_name} does not exist"
    except exceptions.MilvusException as e:
        return f"Failed to drop collection {collection_name}: {e}"

def check_files_in_db(collection_name, file_ids):
    try:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            expr = f"id in [{', '.join(map(str, file_ids))}]"
            result = collection.query(expr, output_fields=["id"])
            existing_ids = [res["id"] for res in result]
            missing_ids = list(set(file_ids) - set(existing_ids))
            return existing_ids, missing_ids
        else:
            return [], file_ids
    except exceptions.MilvusException as e:
        return [], file_ids


# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, exceptions

# def search_data(collection, embedding, top_k):
#     collection.load()  # Ensure collection is loaded before searching
#     search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
#     results = collection.search([embedding], "embedding", search_params, limit=top_k, output_fields=["file_name"])
#     return results

# def connect_milvus():
#     connections.connect("default", host="localhost", port="19530")

# def create_collection(collection_name):
#     if utility.has_collection(collection_name):
#         print(f"Collection {collection_name} already exists")
#         collection = Collection(collection_name)
#         collection.load()
#         return collection 
#     fields = [
#         FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
#         FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
#         FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
#         FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=6)  # "need" or "noneed"
#     ]
#     schema = CollectionSchema(fields, "Object Embeddings")
#     collection = Collection(collection_name, schema)
#     index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
#     collection.create_index("embedding", index_params)
#     collection.load()
#     return collection

# def insert_data(collection, ids, embeddings, file_names, sub_object_type):
#     id = [ids]
#     path = [file_names]
#     types = [sub_object_type]
#     data = [id, embeddings, path, types]
#     collection.insert(data)
#     collection.load()  # Ensure collection is loaded after insertion

# def list_collections():
#     return utility.list_collections()

# def drop_collection(collection_name):
#     try:
#         if utility.has_collection(collection_name):
#             collection = Collection(collection_name)
#             collection.drop()
#             return f"Collection {collection_name} dropped successfully"
#         else:
#             return f"Collection {collection_name} does not exist"
#     except exceptions.MilvusException as e:
#         return f"Failed to drop collection {collection_name}: {e}"

# def check_files_in_db(collection_name):
#     try:
#         if utility.has_collection(collection_name):
#             collection = Collection(collection_name)
#             expr_need = 'type == "need"'
#             expr_noneed = 'type == "noneed"'
            
#             result_need = collection.query(expr_need, output_fields=["file_name"])
#             result_noneed = collection.query(expr_noneed, output_fields=["file_name"])
            
#             existing_files_need = [res["file_name"] for res in result_need]
#             existing_files_noneed = [res["file_name"] for res in result_noneed]
            
#             return existing_files_need, existing_files_noneed
#         else:
#             return [], []
#     except exceptions.MilvusException as e:
#         return [], []
