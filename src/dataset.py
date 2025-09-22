import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class UrbanSentimentDataset(Dataset):
    def __init__(self, metadata_file: str, image_dir: str, config: dict, split: str = 'train'):
        """
        Args:
            metadata_file (str): Шлях до .csv файлу (train, validation, або test).
            image_dir (str): Шлях до директорії з усіма зображеннями.
            config (dict): Словник з конфігурацією.
            split (str): Тип набору даних ('train', 'validation', 'test').
        """
        self.metadata = pd.read_csv(metadata_file)
        self.image_dir = image_dir
        self.config = config
        self.split = split

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(config['model']['classes'])}

        self.transform_for_model = transforms.Compose([
            transforms.Resize((config['preprocessing']['image_size'], config['preprocessing']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['preprocessing']['mean'], std=config['preprocessing']['std'])
        ])

        self.transform_for_visualization = transforms.Compose([
            transforms.Resize((config['preprocessing']['image_size'], config['preprocessing']['image_size'])),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        """Повертає загальну кількість семплів у датасеті."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        """Завантажує та повертає один семпл з датасету за його індексом."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]

        img_filename = row['image_path'].replace('images/', '')
        img_path = os.path.join(self.image_dir, img_filename)

        try:
            image_pil = Image.open(img_path).convert('RGB')
            processed_image = self.transform_for_model(image_pil)
            original_image_tensor = self.transform_for_visualization(image_pil)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Using a placeholder.")
            img_size = self.config['preprocessing']['image_size']
            processed_image = torch.zeros((3, img_size, img_size))
            original_image_tensor = torch.zeros((3, img_size, img_size))

        text = row['text_processed']
        geo_coords = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float)
        label_name = row['sentiment_label']
        label_idx = self.class_to_idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)
        sample_id = row['sample_id']

        sample = {
            'image': processed_image,
            'original_image_tensor': original_image_tensor,
            'text': text,
            'geo': geo_coords,
            'label': label,
            'sample_id': sample_id
        }

        return sample