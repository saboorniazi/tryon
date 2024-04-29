# from fastapi import FastAPI, File, UploadFile
# from PIL import Image
# import io
# import torch
# from pathlib import Path
# import os
# import sys
# PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
# sys.path.insert(0, str(PROJECT_ROOT))
# from utils_ootd import get_mask_location
# from preprocess.openpose.run_openpose import OpenPose
# from preprocess.humanparsing.run_parsing import Parsing
# from ootd.inference_ootd_hd import OOTDiffusionHD
# from ootd.inference_ootd_dc import OOTDiffusionDC
# openpose_model_hd = OpenPose(0)
# parsing_model_hd = Parsing(0)
# ootd_model_hd = OOTDiffusionHD(0)

# openpose_model_dc = OpenPose(1)
# parsing_model_dc = Parsing(1)
# ootd_model_dc = OOTDiffusionDC(1)


# category_dict = ['upperbody', 'lowerbody', 'dress']
# category_dict_utils = ['upper_body', 'lower_body', 'dresses']


# example_path = os.path.join(os.path.dirname(__file__), 'examples')
# # model_hd = os.path.join(example_path, 'model/model_1.png')
# # garment_hd = os.path.join(example_path, 'garment/03244_00.jpg')
# model_dc = os.path.join(example_path, 'model/model_8.png')
# garment_dc = os.path.join(example_path, 'garment/048554_1.jpg')
# def process_dc(vton_img, garm_img, category, n_samples, n_steps, image_scale, seed):
#     model_type = 'dc'
#     if category == 'Upper-body':
#         category = 0
#     elif category == 'Lower-body':
#         category = 1
#     else:
#         category =2

#     with torch.no_grad():
#         garm_img = Image.open(garm_img).resize((768, 1024))
#         vton_img = Image.open(vton_img).resize((768, 1024))
#         keypoints = openpose_model_dc(vton_img.resize((384, 512)))
#         model_parse, _ = parsing_model_dc(vton_img.resize((384, 512)))

#         mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
#         mask = mask.resize((768, 1024), Image.NEAREST)
#         mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
#         masked_vton_img = Image.composite(mask_gray, vton_img, mask)

#         images = ootd_model_dc(
#             model_type=model_type,
#             category=category_dict[category],
#             image_garm=garm_img,
#             image_vton=masked_vton_img,
#             mask=mask,
#             image_ori=vton_img,
#             num_samples=n_samples,
#             num_steps=n_steps,
#             image_scale=image_scale,
#             seed=seed,
#         )

#     return images

# app = FastAPI()
# # @app.post("/process_hd")
# # async def process_hd(vton_img: UploadFile = File(...), garm_img: UploadFile = File(...),
# #                      n_samples: int = 1, n_steps: int = 20, image_scale: float = 2.0, seed: int = -1):
# #     model_type = 'hd'
# #     category = 0
# #     vton_img = Image.open(io.BytesIO(await vton_img.read())).resize((768, 1024))
# #     garm_img = Image.open(io.BytesIO(await garm_img.read())).resize((768, 1024))
# #     # Rest of the code is similar to your original process_hd function
# #     return {"message": "Processed images"}
# @app.post("/process_dc")
# async def process_dc(vton_img: UploadFile = File(...), garm_img: UploadFile = File(...),
#                      category: str = 'Upper-body', n_samples: int = 1, n_steps: int = 20,
#                      image_scale: float = 2.0, seed: int = -1):
#     model_type = 'dc'
#     if category == 'Upper-body':
#         category = 0
#     elif category == 'Lower-body':
#         category = 1
#     else:
#         category = 2
#     vton_img = Image.open(io.BytesIO(await vton_img.read())).resize((768, 1024))
#     garm_img = Image.open(io.BytesIO(await garm_img.read())).resize((768, 1024))
#     # Rest of the code is similar to your original process_dc function
#     return {"message": "Processed images"}

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=5033)

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io 
import torch
from pathlib import Path
import os
import sys
import boto3
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import time
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


openpose_model_hd = OpenPose(0)
parsing_model_hd = Parsing(0)
ootd_model_hd = OOTDiffusionHD(0)

openpose_model_dc = OpenPose(0)
parsing_model_dc = Parsing(0)
ootd_model_dc = OOTDiffusionDC(0)

load_dotenv()
# Initialize boto3 client for S3
s3_client = boto3.client('s3')


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

example_path = os.path.join(os.path.dirname(__file__), 'examples')
model_dc = os.path.join(example_path, 'model/model_8.png')
garment_dc = os.path.join(example_path, 'garment/048554_1.jpg')

app = FastAPI()

import requests
from io import BytesIO
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        return None


def upload_image_to_s3(image_bytes, bucket_name, file_name):
    try:
        # Upload the image to S3 bucket
        # s3_client.upload_fileobj(io.BytesIO(image_bytes), bucket_name, file_name)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"uploads/{file_name}",
            Body=image_bytes
        )
        # Get the URL of the uploaded image
        image_url = f"https://{bucket_name}.s3.ap-south-1.amazonaws.com/uploads/{file_name}"
        return image_url
    except Exception as e:
        # Handle upload error
        print("Error uploading image to S3:", e)
        return None

def process_dc(vton_img_url, garm_img_url, category, n_samples, n_steps, image_scale, seed, bucket_name, file_prefix):
    model_type = 'dc'
    if category == 'Upper-body':
        category = 0
    elif category == 'Lower-body':
        category = 1
    else:
        category = 2

    vton_img = download_image(vton_img_url)
    garm_img = download_image(garm_img_url)

    if vton_img is None or garm_img is None:
        return {"error": "Failed to download images"}

    with torch.no_grad():
        vton_img = Image.open(vton_img).resize((768, 1024))
        garm_img = Image.open(garm_img).resize((768, 1024))
        keypoints = openpose_model_dc(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_dc(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_dc(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

        # Convert the resulting image to bytes
        image_bytes = io.BytesIO()
        images[0].save(image_bytes, format='JPEG')  # Convert to bytes
        image_bytes.seek(0)

        # Upload the image to S3 bucket
        file_name = f"{file_prefix}.jpg"  # Change the file extension as needed
        image_url = upload_image_to_s3(image_bytes.getvalue(), bucket_name, file_name)

    return {"image_url": image_url}

@app.post("/process_dc")
async def process_dc_endpoint(vton_img:str, garm_img:str,category: str = 'Upper-body' ):
    seed: int = -1
    image_scale: float = 0.1
    n_steps: int = 20
    n_samples: int = 1
    bucket_name = 'pricerpics'
    file_prefix='result'
    result = process_dc(vton_img, garm_img,
                        category, n_samples, n_steps, image_scale, seed,bucket_name,file_prefix)
    print(result)
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)
    # return result
# Define your AWS S3 bucket name
BUCKET_NAME = 'pricerpics'

# Define your AWS S3 bucket folder
BUCKET_FOLDER = 'uploads'
@app.post("/upload_to_s3")
async def upload_to_s3(file: UploadFile = File(...)):
    # Read file contents
    file_contents = await file.read()
    
    # Upload file to S3
    try:
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=f"{BUCKET_FOLDER}/{file.filename}",
            Body=file_contents
        )
        image_url = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_S3_REGION_NAME')}.amazonaws.com/uploads/{file.filename}"
        
        return {"message": f"File '{file.filename}' uploaded successfully to S3 bucket '{BUCKET_NAME}'", "image_url": image_url}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # import nest_asyncio
    # from pyngrok import ngrok
    # auth_token = "2UhHVTke0jHjM27lmW6IlvIbObe_59DPt4e3QsayiKvD6vXr7"

    # # Set the authtoken
    # ngrok.set_auth_token(auth_token)

    # # Connect to ngrok
    # ngrok_tunnel = ngrok.connect(5033)

    # # Print the public URL
    # print('Public URL:', ngrok_tunnel.public_url)

    # # Apply nest_asyncio
    # nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=5033)
