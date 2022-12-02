# Created by shaji at 02.12.2022
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def mask_rcnn(test_img, weights=None):
    if weights is None:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    img_tensor_int = pil_to_tensor(Image.open(test_img)).unsqueeze(dim=0)
    img_tensor_float = img_tensor_int / 255.0
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    img_preds = model(img_tensor_float)
    img_preds[0]["boxes"] = img_preds[0]["boxes"][img_preds[0]["scores"] > 0.8]
    img_preds[0]["labels"] = img_preds[0]["labels"][img_preds[0]["scores"] > 0.8]
    img_preds[0]["masks"] = img_preds[0]["masks"][img_preds[0]["scores"] > 0.8]
    img_preds[0]["scores"] = img_preds[0]["scores"][img_preds[0]["scores"] > 0.8]

    categories = weights.meta["categories"]
    img_labels = img_preds[0]["labels"].numpy()
    img_annot_labels = [f"{categories[label]}: {prob:.2f}" for label, prob in
                                  zip(img_labels, img_preds[0]["scores"].detach().numpy())]
    img_output_tensor = draw_bounding_boxes(image=img_tensor_int[0],
                                     boxes=img_preds[0]["boxes"],
                                     labels=img_annot_labels,
                                     colors=["red" if categories[label] == "person" else "green" for label in img_labels],
                                     width=2)

    img_masks_float = img_preds[0]["masks"].squeeze(1)
    img_masks_float[img_masks_float < 0.8] = 0
    img_masks_bool = img_masks_float.bool()
    img_output_tensor = draw_segmentation_masks(img_output_tensor, masks=img_masks_bool, alpha=0.8)
    img_output = to_pil_image(img_output_tensor)

    return img_output
