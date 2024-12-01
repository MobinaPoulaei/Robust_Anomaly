import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils.cifar_new import CIFARNew 
from utils.tiny_imagenet import TinyImageNetDataset
from utils.tiny_imagenet_c import TinyImageNetCDataset
from utils.mvtec import MVTecDataset
from utils.transforms import *

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transforms(image_size=224):
    CLIP_TRANSFORMS = transforms.Compose([
                                    transforms.Resize(image_size, interpolation=BICUBIC),
                                    transforms.CenterCrop(image_size),
                                    _convert_image_to_rgb,
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                        (0.26862954, 0.26130258, 0.27577711)),
                                    ])

    MASK_TRASFORM = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
                ])
    return CLIP_TRANSFORMS, MASK_TRASFORM

CIFAR_COMMON_CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                            'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 
                            'elastic_transform', 'pixelate', 'jpeg_compression']

def prepare_data(dataset, data_dir, corruption, batch_size=128, num_workers=1, image_size=224):

    """
    Prepare the specified dataset.

    Parameters:
    ----------
    dataset : str
        The name of the dataset to prepare. Should be one of 'cifar10', 'cifar100', 'tiny-imagenet', 'visda', 'PACS', 'office_home', 'VLCS'.
    data_dir : str
        The root directory where the dataset is stored.
    corruption : str
        The type of corruption to apply to the dataset, if applicable. 
        Only used for 'cifar10', 'cifar100', and 'tiny-imagenet'.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.
o   pbect_name:for mvtec
    Returns:
    -------
    tuple
        A tuple containing:
        - loader (torch.utils.data.DataLoader): DataLoader for the prepared dataset.
        - dataset (torchvision.datasets or ImageFolder): The prepared dataset.

    """


    if dataset == 'cifar10':
        loader, classes = prepare_cifar10_data(data_dir, corruption, batch_size=batch_size, num_workers=num_workers)

    elif dataset in  ["visda", "PACS", "office_home", "VLCS"]:
        loader, classes = prepare_imagefolder_data(data_dir, dataset, batch_size=batch_size, num_workers=num_workers)
    elif dataset in ['mvtec']:
        loader, classes = prepare_mvtec_data(data_dir, corruption, batch_size=batch_size, num_workers=num_workers, image_size=image_size)
    elif dataset in ['miad']:
        loader, classes = prepare_miad_data(data_dir, corruption, batch_size=batch_size, num_workers=num_workers, image_size=image_size)

    else:
        raise Exception(f'Dataset {dataset} not found/implemented!')
    
    return loader, classes





def prepare_cifar10_data(data_dir, corruption, batch_size=128, level=5, size=10000, num_workers=1, image_size=224):
    """
    A function to prepare different versions of CIFAR-10 dataset.

    Parameters:
    ----------
    data_dir : str
        The root directory where the dataset is stored.
    corruption : str
        The type of corruption to apply to the dataset.
        Should be 'original', one of CIFAR_COMMON_CORRUPTIONS, or 'cifar_new'.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    level : int, optional
        The severity level of corruption for CIFAR-10-C. Default is 5.
    size : int, optional
        The number of images to select per corruption level for CIFAR-10-C. Default is 10000.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.

    Returns:
    -------
    tuple
        A tuple containing:
        - loader (torch.utils.data.DataLoader): DataLoader for the prepared dataset.
        - dataset (torchvision.datasets): The prepared dataset.

    """
    CLIP_TRANSFORMS, MASK_TRASFORM = get_transforms(image_size=image_size)
    print(corruption)
    if corruption == 'original':
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=CLIP_TRANSFORMS)
        print(f"The original CIFAR-10 dataset is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")
        
    elif corruption in CIFAR_COMMON_CORRUPTIONS:
        dataset_raw = np.load(data_dir + '/CIFAR-10-C/%s.npy' % (corruption))
        dataset_raw = dataset_raw[(level - 1)*size: level*size]
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=CLIP_TRANSFORMS)
        dataset.data = dataset_raw
        print(f"The CIFAR-10-C dataset with '{corruption}' corruption is selected, number of images: {len(dataset)}!\ndatadir: {data_dir + '/CIFAR-10-C'}")

    elif corruption == 'cifar_new':
        dataset = CIFARNew(root=data_dir + '/CIFAR-10.1/', transform=CLIP_TRANSFORMS)
        print(f"The CIFAR-10.1 (new CIFAR10 test data) dataset is selected, number of images: {len(dataset)}!\ndatadir: {data_dir + '/CIFAR-10.1/'}")

    else:
        raise Exception(f'Corruption {corruption} not found!')


    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    classes = dataset.classes
    return loader, classes



def prepare_cifar100_data(data_dir, corruption, batch_size=128, level=5, size=10000, num_workers=1, image_size=224): 
    """
    A function to prepare different versions of CIFAR-100 dataset.

    Parameters:
    ----------
    data_dir : str
        The root directory where the dataset is stored.
    corruption : str
        The type of corruption to apply to the dataset. 
        Should be 'original' or one of CIFAR_COMMON_CORRUPTIONS.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    level : int, optional
        The severity level of corruption for CIFAR-100-C. Default is 5.
    size : int, optional
        The number of images to select per corruption level for CIFAR-100-C. Default is 10000.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.

    Returns:
    -------
    tuple
        A tuple containing:
        - loader (torch.utils.data.DataLoader): DataLoader for the prepared dataset.
        - dataset (torchvision.datasets.CIFAR100): The prepared dataset.

    """

    CLIP_TRANSFORMS, MASK_TRASFORM = get_transforms(image_size=image_size)
    if corruption == 'original':
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=CLIP_TRANSFORMS)
        print(f"The original CIFAR-100 dataset is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")

    elif corruption in CIFAR_COMMON_CORRUPTIONS:
        dataset_raw = np.load(data_dir + '/CIFAR-100-C/%s.npy' % (corruption))
        dataset_raw = dataset_raw[(level - 1)*size: level*size]
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=CLIP_TRANSFORMS)
        dataset.data = dataset_raw
        print(f"The CIFAR-100-C dataset with '{corruption}' corruption is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")
    
    else:
        raise Exception(f'Corruption {corruption} not found!')

  
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    classes = dataset.classes
    return loader, classes



def prepare_tinyimagenet(data_dir, corruption, batch_size=128, level=5, num_workers=1, image_size=224):
    """
    Prepare different versions of the TinyImageNet dataset.

    Parameters:
    ----------
    data_dir : str
        The root directory where the dataset is stored.
    corruption : str
        The type of corruption to apply to the dataset.
        Should be 'original' or one of CIFAR_COMMON_CORRUPTIONS.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    level : int, optional
        The severity level of corruption for Tiny-ImageNet-C. Default is 5.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.

    Returns:
    -------
    tuple
        A tuple containing:
        - loader (torch.utils.data.DataLoader): DataLoader for the prepared dataset.
        - dataset (TinyImageNetDataset or TinyImageNetCDataset): The prepared dataset.

    
    """
    CLIP_TRANSFORMS, MASK_TRASFORM = get_transforms(image_size=image_size)
    if corruption == 'original':
        dataset = TinyImageNetDataset(data_dir + '/tiny-imagenet-200/', mode='val', transform=CLIP_TRANSFORMS)
        print(f"The original TinyImageNetDataset dataset is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")
    
    elif corruption in CIFAR_COMMON_CORRUPTIONS:
        dataset = TinyImageNetCDataset(data_dir + '/Tiny-ImageNet-C/', corruption=corruption, level=level, transform=CLIP_TRANSFORMS)
        print(f"The TinyImageNetDataset dataset with '{corruption}' corruption is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    classes = dataset.classes
    return loader, classes


def prepare_imagefolder_data(data_dir, name, batch_size=128, num_workers=1, image_size=224):

    """
    Prepare an image dataset from a directory using ImageFolder.

    Parameters:
    ----------
    data_dir : str
        The root directory where the dataset is stored.
    name : str
        The name of the dataset.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.

    Returns:
    -------
    tuple
        A tuple containing:
        - loader (torch.utils.data.DataLoader): DataLoader for the prepared dataset.
        - dataset (torchvision.datasets.ImageFolder): The prepared dataset.
    """
    CLIP_TRANSFORMS, MASK_TRASFORM = get_transforms(image_size=image_size)
    dataset = ImageFolder(root=data_dir, transform=CLIP_TRANSFORMS)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    print(f"The original {name} dataset is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")

    classes = dataset.classes
    return loader, classes


def prepare_mvtec_data(data_dir, corruption, batch_size, num_workers, image_size=224):
    CLIP_TRANSFORMS, MASK_TRASFORM = get_transforms(image_size=image_size)
    obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    loaders = {}
    for object in obj_list:
        test_dataset = MVTecDataset(root_dir=data_dir, class_name=object, mode='test', transform=CLIP_TRANSFORMS,
                                    mask_transform=MASK_TRASFORM)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        loaders[object] = test_loader
    return loaders, obj_list


def prepare_miad_data(data_dir, corruption, batch_size, num_workers, image_size=224):
    CLIP_TRANSFORMS, MASK_TRASFORM = get_transforms(image_size=image_size)
    obj_list = ['wind_turbine']
    loaders = {}
    for object in obj_list:
        test_dataset = MVTecDataset(root_dir=data_dir, class_name=object, mode='test', transform=CLIP_TRANSFORMS,
                                    mask_transform=MASK_TRASFORM)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        loaders[object] = test_loader
    return loaders, obj_list

def prepare_rayan_data(data_dir, corruption, batch_size, num_workers, image_size=224):
    CLIP_TRANSFORMS, MASK_TRASFORM = get_transforms(image_size=image_size)
    obj_list = ['capsule', 'capsules','juice_bottle','macaroni2', 'pcb3', 'photovoltaic_module']
    loaders = {}
    for object in obj_list:
        test_dataset = MVTecDataset(root_dir=data_dir, class_name=object, mode='test', transform=CLIP_TRANSFORMS,
                                    mask_transform=MASK_TRASFORM)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        loaders[object] = test_loader
    return loaders, obj_list


