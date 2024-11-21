import AnomalyCLIP_lib
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import pandas as pd
import itertools

import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform, get_classnames
import json
import matplotlib.pyplot as plt
torch.cuda.empty_cache()
def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GiB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GiB")
        print(f"  Reserved memory: {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GiB")
        print(f"  Free memory: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 3):.2f} GiB")

print_gpu_memory()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from visualization import visualizer
import matplotlib.pyplot as plt
from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


def tensor_to_numpy(tensor):
    return tensor.permute(1, 2, 0).numpy()


def viz_image(img_tensor):
    img_tensor = img_tensor.squeeze(0)
    to_pil = T.ToPILImage()
    img = to_pil(img_tensor)
    plt.imshow(img)
    plt.axis('off')
    img.save('/home/haghifam/Robust_Anomaly_Detection/AnomalyCLIP/results/img.png')


def store_image_inputs(image, transformed_image, mask, transformed_mask, path):
    cls = path.split('/')[-2]
    filename = path.split('/')[-1]
    object_name = path.split('/')[-4]
    root_dir = os.path.join('/content/drive/MyDrive/AnomalyCLIP_Results/', object_name, cls, filename.split('.')[0])
    os.makedirs(root_dir, exist_ok=True)
    vutils.save_image(image.squeeze(0), root_dir + '/original_image.png')
    vutils.save_image(transformed_image.squeeze(0), root_dir + '/transformed_image.png')
    vutils.save_image(mask.squeeze(0), root_dir + '/original_mask.png')
    vutils.save_image(transformed_mask.squeeze(0), root_dir + '/transformed_mask.png')


def create_single_meta_object(data_path, object_name):
    if os.path.exists(data_path+'/all_meta.json'):
        meta_json_dir = os.path.join(data_path+'/all_meta.json')
    else:
        meta_json_dir = os.path.join(data_path, "meta.json")
    with open(meta_json_dir, 'r') as f:
        data = json.load(f)
    if not os.path.exists(data_path+'/all_meta.json'):
        os.rename(meta_json_dir, data_path + '/all_meta.json')
    object_meta_data = {'train': {object_name: data['train'][object_name]},
                        'test': {object_name: data['test'][object_name]}}
    if os.path.exists(meta_json_dir):
        os.remove(meta_json_dir)
    with open(meta_json_dir, 'w') as file:
        json.dump(object_meta_data, file, indent=4)


def get_run_params(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        raise ValueError("CPU is being used!")
    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
                              "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters, adapted_model_path=args.adapted_weights_path)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform,
                        dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    if args.object_name == "":
        obj_list = test_data.obj_list
    else:
        obj_list = [args.object_name]
    print(obj_list)
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
    model.to(device)
    return (test_dataloader, device, model, text_features, obj_list)


def test(args, iterative=False, run_params=()):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        raise ValueError("CPU is being used!")

    if not iterative:
        test_dataloader, device, model, text_features, obj_list = get_run_params(args)
    else:
        test_dataloader, device, model, text_features, obj_list = run_params

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        #viz_image(image)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']

        # store_image_inputs(image_orig, image, gt_mask_orig, gt_mask, items['img_path'][0])

        if gt_mask == []:
            print("empty gt mask")
            continue
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)

            anomaly_map = anomaly_map.sum(dim=0)
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0)
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            # visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    image_f1_list = []
    image_sensitivity_list = []
    image_specificity_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    pixel_f1_list = []
    pixel_sensitivity_list = []
    pixel_specificity_list = []
    for obj in obj_list:
        print("obj: ", obj)
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        '''
        start_idx = 115 * 7
        end_idx = start_idx + 115
        results[obj]['pr_sp'] = results[obj]['pr_sp'][start_idx:end_idx]
        '''
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            image_f1 = image_level_metrics(results, obj, "image-f1")

            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_f1, pixel_sensitivity, pixel_specificity = pixel_level_metrics(results, obj, "pixel-f1_sensitivity_specificity")

            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(pixel_f1 * 100, decimals=1)))
            table.append(str(np.round(pixel_sensitivity * 100, decimals=1)))
            table.append(str(np.round(pixel_specificity * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            table.append(str(np.round(image_f1 * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            image_f1_list.append(image_f1)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_f1_list.append(pixel_f1)
            pixel_sensitivity_list.append(pixel_sensitivity)
            pixel_specificity_list.append(pixel_specificity)
        table_ls.append(table)

    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean',
                         str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                         ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                         str(np.round(np.mean(pixel_f1_list) * 100, decimals=1)),
                         str(np.round(np.mean(pixel_sensitivity_list) * 100, decimals=1)),
                         str(np.round(np.mean(pixel_specificity_list) * 100, decimals=1)),
                         str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(image_ap_list) * 100, decimals=1)),
                         str(np.round(np.mean(image_f1_list) * 100, decimals=1))
                         ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'pixel_f1', 'pixel_sensitivity',
                           'pixel_specificity', 'image_auroc', 'image_ap', 'image_f1'],
                           tablefmt="pipe")
    logger.info("\n%s", results)
    return (pixel_auroc, pixel_aupro, pixel_f1, pixel_sensitivity, pixel_specificity,
            image_auroc, image_ap, image_f1)


def robustness_test(args):
    brightnesses = [0, 0.2, 1]
    contrasts = [0, 0.5]
    saturations = [0, 0.5]
    hues = [0, 0.5]
    rotations = [0, 40]
    h_flips = [0, 1]
    v_flips = [0, 1]
    noise_mean = [0]
    noise_std = [0]
    columns = ['brightness', 'contrast', 'saturation', 'hue', 'rotation', 'h_flip', 'v_flip', 'noise_mean',
               'noise_std', 'pixel_auroc', 'pixel_aupro', 'pixel_f1', 'pixel_sensitivity', 'pixel_specificity',
               'image_auroc', 'image_ap', 'image_f1']
    os.makedirs(args.save_path, exist_ok=True)
    csv_file = os.path.join(args.save_path, "aug_data_cropped.csv")
    if not os.path.isfile(csv_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)

    combinations = itertools.product(brightnesses, contrasts, saturations, hues, rotations,
                                     h_flips, v_flips, noise_mean, noise_std)
    run_params = get_run_params(args)

    for combination in tqdm(combinations):
        brightness, contrast, saturation, hue, rotation, h_flip, v_flip, noise_mean, noise_std = combination
        args.brightness = brightness
        args.contrast = contrast
        args.saturation = saturation
        args.hue = hue
        args.rotation = rotation
        args.h_flip = h_flip
        args.v_flip = v_flip
        args.noise_mean, args.noise_std = noise_mean, noise_std
        aug_params = {
            'brightness': args.brightness,
            'contrast': args.contrast,
            'saturation': args.saturation,
            'hue': args.hue,
            'rotation': args.rotation,
            'h_flip': args.h_flip,
            'v_flip': args.v_flip,
            'noise_mean': args.noise_mean,
            'noise_std': args.noise_std
        }

        if not ((df[columns[:9]] == pd.Series(aug_params)[columns[:9]]).all(axis=1).any()):
            #try:
            (pixel_auroc, pixel_aupro, pixel_f1, pixel_sensitivity, pixel_specificity, image_auroc, image_ap,
             image_f1) = test(args, iterative=True, run_params=run_params)
            #except Exception as E:
            #    print(f'An error encountered {E}')
            #    continue
            perf_params = {
                'pixel_auroc': pixel_auroc,
                'pixel_aupro': pixel_aupro,
                'pixel_f1': pixel_f1,
                'pixel_sensitivity': pixel_sensitivity,
                'pixel_specificity': pixel_specificity,
                'image_auroc': image_auroc,
                'image_ap': image_ap,
                'image_f1': image_f1
            }
            new_row = {**aug_params, **perf_params}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(csv_file, index=False)
        else:
            print(f"Skipped {args.brightness}, {args.contrast}, "
                  f"{args.saturation}, {args.hue}, {args.rotation}, {args.h_flip}, {args.v_flip}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="/home/haghifam/HDD1/mvtec-ad", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='/home/haghifam/Robust_Anomaly_Detection/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--adapted_weights_path", default=None)
    # augmenstiations:
    parser.add_argument("--enable_transformation", default=False)
    parser.add_argument("--brightness", type=float, default=0)
    parser.add_argument("--contrast", type=float, default=0)
    parser.add_argument("--saturation", type=float, default=0)
    parser.add_argument("--hue", type=float, default=0)
    parser.add_argument("--rotation", type=float, default=0)
    parser.add_argument("--h_flip", type=float, default=0)
    parser.add_argument("--v_flip", type=float, default=0)
    parser.add_argument("--noise_mean", type=float, default=0)
    parser.add_argument("--noise_std", type=float, default=0)
    parser.add_argument("--robustness_test", default=False)
    #object
    parser.add_argument("--object_name", default="")

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    if args.object_name != "":
        create_single_meta_object(args.data_path, args.object_name)
        args.save_path = os.path.join(args.save_path, args.object_name)

    if args.robustness_test:
        robustness_test(args)
    else:
        test(args)
