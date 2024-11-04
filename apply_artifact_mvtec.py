import random

import numpy as np
import cv2
import os
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
from random import randint

def plot_nd_array(array, output_image_path):
    image = Image.fromarray(array)
    image.save(output_image_path)


def extract_background(model, input_image, output_image_path):
    original_height, original_width = input_image.shape[:2]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    mask = output_predictions.byte().cpu().numpy()
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    bin_mask = np.where(mask, 255, 0).astype(np.uint8)

    num_labels, labels_im = cv2.connectedComponents(bin_mask)

    for label in range(1, num_labels):
        component_mask = (labels_im == label).astype(np.uint8) * 255
        zero_count = np.count_nonzero(component_mask == 0)
        total_count = component_mask.size

        if zero_count / total_count < 1:
            bin_mask[labels_im == label] = 255
    background_mask = 255 - bin_mask

    plot_nd_array(background_mask, output_image_path)

    return background_mask


def place_artifacts_in_background(input_image, artifact_image, background_mask, num_objects=5):
    positions = np.where(background_mask == 255)
    for i in range(positions[0].shape[0]):
        x, y = positions[0][i], positions[1][i]
        if not ((x < 80 or x > 900) and (y < 80 or y > 900)):
            if random.random() < 0.5:
                positions[0][i], positions[1][i] = randint(0, 80), randint(0, 1024)
            else:
                positions[0][i], positions[1][i] = randint(900, 1000), randint(0, 80)

    if len(positions[0]) == 0:
        print("No background area found")
        return input_image

    h, w = input_image.shape[:2]

    for _ in range(num_objects):
        artifact_resized = cv2.resize(artifact_image, (random.randint(50, 100), random.randint(50, 100)))
        obj_h, obj_w = artifact_resized.shape[:2]
        random_index = randint(0, len(positions[0]) - 1)
        y, x = positions[0][random_index], positions[1][random_index]

        while y + obj_h > h or x + obj_w > w:
            random_index = random.randint(0, len(positions[0]) - 1)
            y, x = positions[0][random_index], positions[1][random_index]

        roi = input_image[y:y+obj_h, x:x+obj_w]

        gray_artifact = cv2.cvtColor(artifact_resized, cv2.COLOR_BGR2GRAY)
        _, artifact_mask = cv2.threshold(gray_artifact, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(artifact_mask)

        image_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        artifact_fg = cv2.bitwise_and(artifact_resized, artifact_resized, mask=artifact_mask)

        combined = cv2.add(image_bg, artifact_fg)
        input_image[y:y+obj_h, x:x+obj_w] = combined

    return input_image


def apply_artifacts(mvtec_path, output_path, artificial_object_path_normal, artificial_object_path_anomaly):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    artificial_object_normal = cv2.imread(artificial_object_path_normal)
    artificial_object_anomaly = cv2.imread(artificial_object_path_anomaly)


    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    for class_name in os.listdir(mvtec_path):
        class_folder = os.path.join(mvtec_path, class_name, 'test')
        if not os.path.isdir(class_folder):
            continue
        for defect_type in os.listdir(class_folder):
            defect_folder = os.path.join(class_folder, defect_type)
            if defect_folder == 'good':
                artificial_object = artificial_object_normal
            else:
                artificial_object = artificial_object_anomaly

            defect_output_folder = os.path.join(output_path, class_name, 'test', defect_type)
            os.makedirs(defect_output_folder, exist_ok=True)

            for image_name in os.listdir(defect_folder):
                output_image_path = os.path.join(defect_output_folder, image_name)
                temp_output_path = os.path.join(defect_output_folder, "bg_"+image_name)
                image_path = os.path.join(defect_folder, image_name)
                image = cv2.imread(image_path)
                background_mask = extract_background(model, image, temp_output_path)
                output_image = place_artifacts_in_background(image, artificial_object, background_mask,
                                                             num_objects=randint(2, 6))
                cv2.imwrite(output_image_path, output_image)


if __name__== '__main__':
    mvtec_path = '/home/haghifam/HDD1/mvtec-ad'
    output_path = '/home/haghifam/Datasets/mvtec_artifacts'
    artificial_object_path = '/home/haghifam/Datasets/artifact.JPEG'
    artificial_object_path_anomaly = "/home/haghifam/Datasets/artifact2.png"
    apply_artifacts(mvtec_path, output_path, artificial_object_path, artificial_object_path_anomaly)
