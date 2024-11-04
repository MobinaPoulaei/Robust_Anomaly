import torchvision.transforms as transforms
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from AnomalyCLIP_lib.transform import image_transform
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
import torchvision.transforms as T
import torch
from PIL import Image
import json
import os


class GaussianNoiseTransform:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = T.ToTensor()(img)
            noise = torch.randn(img.size()) * self.std + self.mean
            img = img + noise
            img = T.ToPILImage()(img.clamp(0, 1))
            return img
        else:
            raise TypeError("Input should be a PIL.Image.Image instance.")


def get_classnames(file_path):
    with open(os.path.join(file_path, 'meta.json'), 'r') as f:
        data = json.load(f)
    train_data = data.get('train', {})
    classes = train_data.keys()
    return list(classes)


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def get_augmentation_transform(transformation_params):
    brightness = transformation_params['brightness']
    contrast =  transformation_params['contrast']
    saturation = transformation_params['saturation']
    hue = transformation_params['hue']
    degree = transformation_params['degree']
    h_flip_prob = transformation_params['h_flip_prob']
    v_flip_prob = transformation_params['v_flip_prob']
    noise_mean = transformation_params['noise_mean']
    noise_std = transformation_params['noise_std']
    spatial_augmentation_transform = T.Compose([
        T.RandomRotation(degrees=(degree, degree)),
        T.RandomHorizontalFlip(p=h_flip_prob),
        T.RandomVerticalFlip(p=v_flip_prob)
    ])
    non_spatial_transform = T.Compose([
        T.ColorJitter(brightness=brightness,
                      contrast=contrast,
                      saturation=saturation,
                      hue=hue),
        GaussianNoiseTransform(mean=noise_mean, std=noise_std)
    ])
    augmentation_transform = T.Compose([
        *non_spatial_transform.transforms,
        *spatial_augmentation_transform.transforms
    ])
    return augmentation_transform, spatial_augmentation_transform


def get_transform(args):
    if args.enable_transformation:
        augmentation_transform, spatial_augmentation_transform = get_augmentation_transform({
                                                                    'brightness': args.brightness,
                                                                    'contrast': args.contrast,
                                                                    'saturation': args.saturation,
                                                                    'hue': args.hue,
                                                                    'degree': args.rotation,
                                                                    'h_flip_prob': args.h_flip,
                                                                    'v_flip_prob': args.v_flip,
                                                                    'noise_mean': args.noise_mean,
                                                                    'noise_std': args.noise_std
                                                                })
    else:
        augmentation_transform, spatial_augmentation_transform = None, None

    preprocess = image_transform(args.image_size, is_train=False, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD,
                                 arbitrary_transforms=augmentation_transform, enable_arbitrary_transform=args.enable_transformation)
    target_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.CenterCrop(args.image_size),
        T.ToTensor()
    ])
    if args.enable_transformation:
        target_transform = T.Compose([
            *spatial_augmentation_transform.transforms,
            *target_transform.transforms
        ])

    preprocess.transforms[0] = T.Resize(size=(args.image_size, args.image_size),
                                                 interpolation=transforms.InterpolationMode.BICUBIC,
                                                 max_size=None, antialias=None)
    preprocess.transforms[1] = T.CenterCrop(size=(args.image_size, args.image_size))

    return preprocess, target_transform
