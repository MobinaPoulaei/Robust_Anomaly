import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pytorch_grad_cam.utils.image import show_cam_on_image
import random

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from logger import get_logger
from utils import get_transform
from PIL import Image

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_nd_array(array, output_image_path):
    image = Image.fromarray(array)
    image.save(output_image_path)

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    # Load model
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    # Load dataset
    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # Load prompt learner and checkpoint
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

    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]  # Get class name
        img_path = items['img_path'][0]  # Assuming 'img_path' contains the relative path within the dataset
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        # Get attention weights
        _ = model.encode_image(image, features_list, DPAM_layer=20)
        attn_weights = model.visual.transformer.resblocks[-1].attn_weights
        avg_attn_weights = attn_weights.mean(dim=1)
        cls_attn_map = avg_attn_weights[0, -1, 1:]
        num_patches = int(cls_attn_map.size(0) ** 0.5)
        cls_attn_map = cls_attn_map.reshape(num_patches, num_patches).cpu().numpy()

        # Convert image tensor to numpy
        image_np = image[0].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        image_np = np.uint8(255 * image_np)

        # Resize the attention map
        attention_map_resized = cv2.resize(cls_attn_map, (image_np.shape[1], image_np.shape[0]))
        attention_map_resized = (attention_map_resized - np.min(attention_map_resized)) / (np.max(attention_map_resized) - np.min(attention_map_resized))

        attention_map_resized = np.uint8(255 * attention_map_resized)
        plot_nd_array(attention_map_resized, "/home/haghifam/Robust_Anomaly_Detection/attn_map_grey.jpeg")

        # Apply a colormap to the attention map
        heatmap = cv2.applyColorMap(attention_map_resized, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay the heatmap on the original image
        overlay_image = cv2.addWeighted(image_np, 0.5, heatmap, 0.5, 0)

        # Determine the relative path within the dataset
        relative_path = os.path.relpath(img_path, args.data_path)
        save_dir = os.path.join(args.save_path, 'att_map', os.path.dirname(relative_path))
        os.makedirs(save_dir, exist_ok=True)

        # Define the output file name
        output_filename = os.path.join(save_dir, f'{os.path.splitext(os.path.basename(img_path))[0]}_attention_map.png')
        print(output_filename)

        # Plotting and adding the color bar
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay_image)
        plt.axis('off')
        plt.title(f'Attention Map Overlay - {cls_name}')

        # Add a color bar
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=0, vmax=255)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, orientation='vertical', fraction=0.046, pad=0.04)

        # Save the visualization with color bar
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="/home/haghifam/Datasets/mvtec_artifacts", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='/home/haghifam/Robust_Anomaly_Detection/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth')

    # model
    parser.add_argument("--dataset", default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="model depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="number of context tokens")
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
