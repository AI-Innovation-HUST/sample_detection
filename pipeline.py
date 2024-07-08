import os
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import clip
from PIL import Image
import os
import random
import yaml
import time
from models import *
from PIL import Image
import cv2
import skimage.measure as sm

def predict(img,top_k):
    input=preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        tik=time.time()
        image_features = model.encode_image(input)
        print("Time processing:",time.time()-tik)
        image_features=torch.squeeze(image_features,0)
        output=image_features.detach().cpu().numpy()
        output=output.tolist()
        hits = qdrant.search(
        collection_name=config["collection_name"],
        query_vector=output,
        limit=top_k
        )   
        images=[]
        for hit in hits:
            if hit.score > config["Threshold"]:
                images.append(Image.open(hit.payload["image_name"]))
        return images
class Pipeline():
    def __init__(self,model_path=None):
        self.segmentation = ISNetDIS()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.segmentation.load_state_dict(torch.load(model_path,map_location=self.device))
        self.segmentation.eval()
        self.input_size=[1024,1024]
        self.segmentation.to(self.device)
        self.model_emb, self.preprocess = clip.load("ViT-B/32", device=self.device)

        
    def preprocess_seg(self,image:np.ndarray):
        # image = Image.fromarray(image).convert("RGB")
        im_tensor = torch.tensor(image, dtype=torch.float32).permute(2,0,1).to(self.device)
        im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), self.input_size, mode="bilinear").type(torch.uint8)
        image_ = torch.divide(im_tensor,255.0)
        image_ = normalize(image_,[0.5,0.5,0.5],[1.0,1.0,1.0])
        return image_
    def extract_emb(self,image:np.ndarray):
        # Segmentation object
        im_shp=image.shape[0:2]
        input = self.preprocess_seg(image)
        input=input.to(self.device)
        result=self.segmentation(input)
        result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        mask = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
        cropped_image = cv2.bitwise_and(image, image, mask=mask)

        # Crop object in image
        # gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # largest_contour = max(contours, key=cv2.contourArea)
        # (x, y, w, h) = cv2.boundingRect(largest_contour)
        # image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

        # object_cropped = image[y:y+h,x:x+w]
        # cv2.imwrite("crop.jpg",cropped_image)
        # Emb object
        object_cropped = Image.fromarray(cropped_image).convert("RGB")
        input=self.preprocess(object_cropped).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model_emb.encode_image(input)
            image_features=torch.squeeze(image_features,0)
            output=image_features.detach().cpu().numpy()
            output=output.tolist() # dim 512
        torch.cuda.empty_cache()        
        return output
    
# if __name__ == "__main__":
#     pipe = Pipeline(model_path="saved_models/isnet-general-use.pth")
#     img = cv2.imread("/home/truongan/sample_detection/AI_test/1/2024_06_26_17_24_IMG_1592.JPG")
#     output = pipe.extract_emb(img)
#     print((output))
    
            
        
    
