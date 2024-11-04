import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform

# Additional imports for Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import matplotlib.pyplot as plt


def vit_reshape_transform(tensor):
    tensor = tensor[1:, :, :]  # Now shape: [num_patches, batch_size, embedding_dim]

    # Determine the spatial dimensions (assuming patches form a square grid)
    num_patches = tensor.shape[0]  # Number of patches
    spatial_dim = int(num_patches ** 0.5)  # Assume patches form a square grid

    # Permute the tensor to [batch_size, embedding_dim, num_patches]
    tensor = tensor.permute(1, 2, 0)  # Now shape: [batch_size, embedding_dim, num_patches]

    # Reshape to [batch_size, embedding_dim, height, width]
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], spatial_dim, spatial_dim)

    return tensor

class VisualWrapper(torch.nn.Module):
    def __init__(self, visual_model, features_list, dpam_layer=20):
        super(VisualWrapper, self).__init__()
        self.visual_model = visual_model
        self.features_list = features_list
        self.dpam_layer = dpam_layer

    def forward(self, x):
        image_features, patch_features = self.visual_model(x, self.features_list, DPAM_layer=self.dpam_layer, ori_patch=False,
                                 proj_use=True, ffn=False)
        image_features.requires_grad = True
        image_features.retain_grad = True
        for feature in patch_features:
            feature.requires_grad = True
            feature.retain_grad = True
        return image_features

def grad_hook(grad):
    print(f"Gradient shape: {grad.shape}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    target_layer = model.transformer.resblocks[-1].mlp
    vis_model = VisualWrapper(model.visual, features_list)
    cam = GradCAM(model=vis_model, target_layers=[target_layer], reshape_transform=None)

    print("target:", target_layer)
    for param in target_layer.parameters():
        param.requires_grad = True

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform,
                        dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []}
        metrics[obj] = {'pixel-auroc': 0, 'pixel-aupro': 0, 'image-auroc': 0, 'image-ap': 0}

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Directory to save CAM images
    cam_save_path = os.path.join(args.save_path, 'cam_images')
    os.makedirs(cam_save_path, exist_ok=True)

    model.to(device)
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        vis_model(image)
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())
        # Generate CAM
        image.requires_grad = True
        image.retain_grad = True

        grayscale_cam = cam(input_tensor=image)
        grayscale_cam = grayscale_cam[0]  # For single image

        # Convert image to numpy
        image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize between 0 and 1

        # Overlay CAM on image
        visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

        # Save the CAM visualization
        cam_filename = f"{cls_name[0]}_{idx}_cam.png"
        cam_filepath = os.path.join(cam_save_path, cam_filename)
        plt.figure(figsize=(10, 10))
        plt.imshow(visualization)
        plt.axis('off')
        plt.savefig(cam_filepath, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # Paths and other arguments
    parser.add_argument("--data_path", type=str, default="/home/haghifam/HDD1/mvtec-ad", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str,
                        default='/home/haghifam/Robust_Anomaly_Detection/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth')
    # Model
    parser.add_argument("--dataset", default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="context tokens")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="text context length")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="feature map layers")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="sigma for Gaussian filter")
    parser.add_argument("--enable_transformation", default=False)

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
