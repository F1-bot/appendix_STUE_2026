import yaml
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from dataset import UrbanSentimentDataset
from models import MultimodalSentimentModel


def set_seed(seed_value):
    """Встановлює seed для відтворюваності результатів."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Виконує одну епоху тренування."""
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        batch_for_device = {
            'image': batch['image'].to(device),
            'text': batch['text'],
            'geo': batch['geo'].to(device)
        }
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(batch_for_device)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """Виконує валідацію або тестування моделі."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch_for_device = {
                'image': batch['image'].to(device),
                'text': batch['text'],
                'geo': batch['geo'].to(device)
            }
            labels = batch['label'].to(device)

            outputs = model(batch_for_device)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    accuracy = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, accuracy, weighted_f1


def main():
    with open('../configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['general']['seed'])

    device = torch.device(config['general']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Використовується пристрій: {device}")

    train_dataset = UrbanSentimentDataset(config['data']['train_csv'], config['data']['image_dir'], config)
    val_dataset = UrbanSentimentDataset(config['data']['validation_csv'], config['data']['image_dir'], config)
    test_dataset = UrbanSentimentDataset(config['data']['test_csv'], config['data']['image_dir'], config)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
                              num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False,
                            num_workers=config['training']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False,
                             num_workers=config['training']['num_workers'])
    print("Датасети та завантажувачі даних створено.")

    model = MultimodalSentimentModel(config).to(device)

    class_counts = train_dataset.metadata['sentiment_label'].value_counts().sort_index()
    class_weights = 1. / torch.tensor(class_counts.values, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(config['model']['classes'])
    class_weights = class_weights.to(device)
    print(f"Розраховані ваги для класів: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'],
                      weight_decay=config['training']['weight_decay'])

    best_f1 = 0.0
    best_model_path = "best_model.pth"

    print("\n--- Початок тренування ---")
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Епоха {epoch + 1}/{config['training']['epochs']} ---")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_f1 = validate(model, val_loader, criterion, device)

        print(f"Втрати на тренуванні: {train_loss:.4f}")
        print(f"Втрати на валідації: {val_loss:.4f}, Точність: {val_accuracy:.4f}, Weighted F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Збережено нову найкращу модель з F1-score: {best_f1:.4f}")

    print("\n--- Завантаження найкращої моделі для фінальної оцінки ---")
    model.load_state_dict(torch.load(best_model_path))

    test_loss, test_accuracy, test_f1 = validate(model, test_loader, criterion, device)

    print("\n--- Результати на тестовому наборі ---")
    print(f"Втрати на тесті: {test_loss:.4f}")
    print(f"Точність на тесті: {test_accuracy:.4f}")
    print(f"Weighted F1-score на тесті: {test_f1:.4f}")
    print("Тренування завершено.")


if __name__ == '__main__':
    main()