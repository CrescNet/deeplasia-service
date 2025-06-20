import logging.config
from utils.log import LOG_CONFIG

logging.config.dictConfig(LOG_CONFIG)

# TODO: add logging to file

from flask import abort, Flask, request, Response
import torch

import numpy as np
import cv2

from fscnn.predict import Predictor as MaskPredictor
from bone_age.models import (
    EfficientModel as BoneAgeModel,
    Predictor as AgePredictor,
    MultiTaskModel as SexModel,
    SexPredictor,
)

import os

app = Flask(__name__)

use_cuda = torch.cuda.is_available()
enable_sex_prediction = True
threads = int(os.getenv('DEEPLASIA_THREADS', 4))

mask_model_path = "./models/fscnn_cos.ckpt"
ensemble = {
    "masked_effnet_super_shallow_fancy_aug": BoneAgeModel(
        "efficientnet-b0",
        pretrained_path="./models/masked_effnet_super_shallow_fancy_aug.ckpt",
        load_dense=True,
    ).eval(),

    "masked_effnet_supShal_highRes_fancy_aug": BoneAgeModel(
        "efficientnet-b0",
        pretrained_path="./models/masked_effnet_supShal_highRes_fancy_aug.ckpt",
        load_dense=True,
    ).eval(),
    "masked_effnet-b4_shallow_pretr_fancy_aug": BoneAgeModel(
        "efficientnet-b4",
        pretrained_path="./models/masked_effnet-b4_shallow_pretr_fancy_aug.ckpt",
        load_dense=True,
    ).eval(),
}
if enable_sex_prediction:
    sex_model_ensemble = {
        "sex_model_mtl": SexModel.load_from_checkpoint(
            "./models/sex_pred_model.ckpt"
        ).eval()
    }

torch.set_num_threads(threads)  # define number of threads for pytorch
mask_predictor = MaskPredictor(checkpoint=mask_model_path, use_cuda=use_cuda)
age_predictor = AgePredictor(ensemble, use_cuda=use_cuda)





def get_prediction(image_bytes, sex, use_mask, use_invChecker, mask_crop=1.15):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)


    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = (img / img.max() * 255).astype(np.uint8)
    if use_mask:
        try:
            mask, vis = mask_predictor(img)
        except Exception as e:
            print("no mask found")
            mask = np.ones_like(img)
            vis = img.copy()

    else:
        mask = None #np.ones_like(img)
        vis = img.copy()

    if mask is not None:
        mask = (mask > mask.max() // 2).astype(np.uint8)

    if use_invChecker and use_mask: #
        img, mask = invChecker(img, mask)

    if sex in ["Male", "male", "m", "M"]:
        sex, sex_input = "m", 1
    elif sex in ["Female", "female", "f", "F", "w", "W"]:
        sex, sex_input = "f", 0

    if sex not in ["m", "f"]:
        raise Exception("Sex is not provided")

    age, stats = age_predictor(img, sex_input, mask=mask, mask_crop=mask_crop)

    return age.item(), sex


def invChecker(img, mask):
    omitExtr = 8
    mask = (mask > mask.max() // 2).astype(np.uint8)  # So that every mask is really a mask
    maskXS = cv2.resize(mask, (100, 100), interpolation=cv2.INTER_AREA)  # To improve speed and single pixel operation effects are independent of Resolution
    maskXS = maskXS[1:-1, 1:-1]  # To cut off eventual Border artefacts without losing too much info
    kernel = np.ones((2, 2), np.uint8)
    maskXSD = cv2.erode(maskXS, kernel, iterations=2)  # dilate(maskXS, kernel, iterations=2)
    maskXSR = cv2.subtract(maskXS, maskXSD).astype(bool)  # create inner border of Hand for comparison

    imgXS = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)  # To improve speed and single pixel operation effects are independent of Resolution
    imgXS = imgXS[1:-1, 1:-1]  # To cut off eventual Border artefacts without losing too much info
    imgXS = cv2.equalizeHist(imgXS * maskXS)

    handR = imgXS[maskXSR]
    handR = handR[(handR < 255 - omitExtr) & (handR > omitExtr)]
    hand = imgXS[maskXS.astype(bool)]
    hand = hand[(hand < 255 - omitExtr) & (hand > omitExtr)]

    if handR.mean() / hand.mean() > 1.2:  # 1.2 found empirically
        img = 255 - img

    return img, mask

@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            abort(400, "No file provided!")

        file = request.files["file"]
        image_bytes = file.read()

        sex = request.form.get("sex")

        use_mask = request.form.get("use_mask", "True") == "True"
        mask_crop = float(request.form.get("mask_crop", 1.15))
        use_invChecker =  request.form.get("use_invChecker", "True") == "True"

        bone_age, sex = get_prediction(image_bytes, sex, use_mask, use_invChecker, mask_crop=mask_crop)

        return {
            "bone_age": bone_age,
            "used_sex": sex,
        }
    except Exception as e:
        logger.exception("Prediction failed")
        abort(500, f"Prediction failed: {str(e)}")

@app.get("/")
def ping():
    with open("deeplasia-api.yml", "r") as f:
        return Response(f.read(), mimetype="application/json")
    abort(404, "Not found!")

if __name__ == "__main__":
    app.run()


# can be called as `python app.py`

# with open("../data/public/Achondroplasia_Slide6.PNG", "rb") as f:
#     image_bytes = f.read()
#     print(get_prediction(image_bytes))

# import requests

# url = "http://localhost:8080/predict"

# test_img = "/home/sebastian/bone2gene/data/public/Achondroplasia_Slide6.PNG"
# files = {'file': open(test_img,'rb')}

# data = {
#     "sex": "female",
#     "use_mask": "1"  # 1 for True, 0 for False
#     "mask_crop":
# }
# resp = requests.post(url, files=files, data=data)
# resp.json()
