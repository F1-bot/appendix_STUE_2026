import yaml
import torch
from torch.utils.data import DataLoader
from dataset import UrbanSentimentDataset
from models import MultimodalSentimentModel


def main():
    with open('../configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['general']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Використовується пристрій: {device}")

    train_dataset = UrbanSentimentDataset(
        metadata_file=config['data']['train_csv'],
        image_dir=config['data']['image_dir'],
        config=config
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )

    print("\n--- Ініціалізація повної мультимодальної моделі ---")
    model = MultimodalSentimentModel(config).to(device)
    print("Модель успішно створено.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Загальна кількість параметрів: {total_params:,}")
    print(f"Кількість параметрів для тренування: {trainable_params:,}")

    model.eval()

    print("\n--- Демонстрація роботи моделі на одному батчі ---")
    sample_batch = next(iter(train_loader))

    batch_for_device = {
        'image': sample_batch['image'].to(device),
        'text': sample_batch['text'],
        'geo': sample_batch['geo'].to(device)
    }

    with torch.no_grad():
        logits = model(batch_for_device)

    print("\n--- Розміри вихідних даних ---")
    print(f"Розмір логітів (сирих прогнозів): {logits.shape}")
    print("\nПриклад логітів для перших 5 семплів:")
    print(logits[:5])


if __name__ == '__main__':
    main()