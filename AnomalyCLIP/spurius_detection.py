import os
from PIL import Image
import torch
import json
import pickle
from transformers import OwlViTProcessor, OwlViTForObjectDetection


def extract_attributes(folder_name, synsets_path):
    fine_tuning_classes =  [folder for folder in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, folder))]
    with open(synsets_path, 'r') as f:
        visual_genome_synsets = json.load(f)
    visual_genome_synsets = list(visual_genome_synsets.keys())
    spurious_attributes = list(set(visual_genome_synsets) - set(fine_tuning_classes))
    return spurious_attributes


def get_all_image_paths_mvtec(main_folder_path):
    image_paths = []
    for class_folder in os.listdir(main_folder_path):
        class_folder_path = os.path.join(main_folder_path, class_folder)

        if os.path.isdir(class_folder_path):
            for subfolder in ['ground_truth', 'test', 'train']:
                subfolder_path = os.path.join(class_folder_path, subfolder)

                if os.path.isdir(subfolder_path):
                    if subfolder == 'train':
                        for train_subfolder in os.listdir(subfolder_path):
                            train_subfolder_path = os.path.join(subfolder_path, train_subfolder)

                            if os.path.isdir(train_subfolder_path):
                                # Collect all image files in the train subfolders
                                for image_file in os.listdir(train_subfolder_path):
                                    if image_file.endswith(('.jpg', '.png')):
                                        image_paths.append(os.path.join(train_subfolder_path, image_file))
                    else:
                        for image_file in os.listdir(subfolder_path):
                            if image_file.endswith(('.jpg', '.png')):
                                image_paths.append(os.path.join(subfolder_path, image_file))

    return image_paths


def detect_spurious_attributes(image_path, model, processor, spurious_attributes, device):
    '''
   image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128), Image.LANCZOS)
    inputs = processor(text=spurious_attributes, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

    detected_spurious = []
    for label, score in zip(labels, scores):
        if score > 0.5:  # Consider detections with confidence > 0.5
            detected_spurious.append(spurious_attributes[label])
    return detected_spurious
    '''
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256), Image.LANCZOS)

    detected_spurious = []

    for i in range(0, len(spurious_attributes), 32):
        batch_attributes = spurious_attributes[i:i + 32]
        inputs = processor(text=batch_attributes, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

        for label, score in zip(labels, scores):
            if score > 0.5:  # Consider detections with confidence > 0.5
                detected_spurious.append(spurious_attributes[label])

    return detected_spurious


def process_spurious(folder_path, synsets_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    all_spurius = []
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    spurious_attributes = extract_attributes(folder_name, synsets_path)
    all_image_paths = get_all_image_paths_mvtec(folder_path)
    for image_path in all_image_paths:
        spurious = detect_spurious_attributes(image_path, model, processor, spurious_attributes, device)
        print(f"Detected spurious attributes in {image_path}: {spurious}")
        all_spurius.append(spurious)
    return all_spurius


if __name__ == '__main__':
    folder_name = '/home/haghifam/HDD1/mvtec-ad'
    synsets_path = r'/home/haghifam/Datasets/object_synsets.json'
    all_spurious = process_spurious(folder_name, synsets_path)
    with open('sp_data.pkl', 'wb') as file:
        pickle.dump(all_spurious, file)