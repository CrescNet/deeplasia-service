from albumentations.pytorch import ToTensorV2
import albumentations as A

from fscnn.lib.models import *
import numpy as np
import torch
import cv2


def get_inference_aug(size=512):
    return A.Compose(
        [
            A.RandomResizedCrop(size, size, scale=(1.0, 1.0), ratio=(1.0, 1.0)), # Outdated
            #A.RandomResizedCrop((size, size), scale=(1.0, 1.0), ratio=(1.0, 1.0)), #For new Version
            ToTensorV2(),
        ],
        additional_targets={"weight": "mask"},
    )


def normalize_image(img: torch.Tensor) -> torch.Tensor:
    img -= img.min()
    return img / img.max()


class Predictor:
    def __init__(
        self,
        size=512,
        checkpoint="/home/rassman/bone2gene/masking/output/version_25/ckp/best_model.ckpt",
        use_cuda=False,
    ):
        self.model = MaskModel.load_from_checkpoint(checkpoint)
        self.model.eval()
        if use_cuda:
            self.model.cuda()
        self.size = size
        self.aug = get_inference_aug(size)

    def __call__(self, raw_image) -> np.array:
        """
        provide a path to an image and the predictor return the extracted hand contour
        """
        h, w = raw_image.shape
        if h > w:
            y = 0
            x = (h - w) // 2
        else:
            x = 0
            y = (w - h) // 2
        image_crop = np.zeros((max(h, w), max(h, w)), np.uint8)  # copy make border

        image_crop[y : y + h, x : x + w] = raw_image
        image = self.aug(image=image_crop)["image"].unsqueeze(dim=0)
        #print("Shape after unsqueeze:", image.shape)
        image = normalize_image(image)
        with torch.no_grad():
            output = self.model(image.to(self.model.device))
        output = torch.softmax(output.squeeze(), dim=0)[1].cpu().numpy()
        output = cv2.resize(
            output, dsize=(max(h, w), max(h, w)), interpolation=cv2.INTER_LINEAR
        )
        binary = (output > 0.25).astype(np.uint8)
        hand, image_crop = self.extract_hand(binary, output, image_crop)
        return (
            hand[y : y + h, x : w + x].copy(),
            image_crop[y : y + h, x : w + x].copy(),
        )

    @staticmethod
    def extract_hand(binary, raw_prediction, org_image):
        conts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        conts = sorted(conts, key=lambda x: cv2.contourArea(x), reverse=True)
        hand = None
        for c in conts:  # extract the largest area containing high confidence
            pred = cv2.drawContours(np.zeros_like(raw_prediction), [c], -1, 1, -1)
            if np.any(pred * (raw_prediction > 0.99)):
                hand = pred
                break
        if hand is None:
            c = conts[0]
            hand = cv2.drawContours(np.zeros_like(raw_prediction), [c], -1, 1, -1)
        hand = hand.astype(np.uint8)
        org_image = (
            org_image / (cv2.bitwise_or(org_image, org_image, mask=hand).max()) * 230
        ).astype(np.uint8)
        org_image = cv2.cvtColor(org_image, cv2.COLOR_GRAY2RGB)
        line_width = (org_image.shape[1] // 250) + 1
        org_image = cv2.drawContours(org_image, [c], -1, (0, 255, 0), line_width)
        return hand, org_image


# def main(
#     input,
#     size=512,
#     checkpoint="/home/rassman/bone2gene/masking/output/version_25/ckp/best_model.ckpt",
#     output="./output/",
#     use_gpu=False,
# ):
#     p = Predictor(size, checkpoint, use_cuda=use_gpu)
#     os.makedirs(output, exist_ok=True)
#     l = (
#         glob(os.path.join(input, "*.png")) + glob(os.path.join(input, "*.jpg"))
#         if os.path.isdir(input)
#         else [input]
#     )
#     for img_path in tqdm(l):
#         hand, vis = p(img_path)
#         img_path = img_path.replace(".jpg", ".png")
#         cv2.imwrite(os.path.join(output, os.path.basename(img_path)), hand)
#         cv2.imwrite(
#             os.path.join(
#                 output, os.path.basename(img_path).replace(".png", "_vis.png")
#             ),
#             vis,
#         )
#
#
# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument(
#         "--checkpoint",
#         default="/home/rassman/bone2gene/masking/output/version_25/ckp/best_model.ckpt",
#     )
#     parser.add_argument(
#         "--input",
#         default="/home/rassman/bone2gene/data/annotated/shox_magdeburg/shox_magd_00001.png",
#         help="can be eiter single image or full directory",
#     )
#     parser.add_argument(
#         "--input_size",
#         default=512,
#         type=int,
#     )
#     parser.add_argument(
#         "--output_dir",
#         default="./output/masks/",
#     )
#     parser.add_argument("--use_gpu", action="store_true")
#
#     args = parser.parse_args()
#
#     main(args.input, args.input_size, args.checkpoint, args.output_dir, args.use_gpu)
