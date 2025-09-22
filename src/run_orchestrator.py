import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataset import UrbanSentimentDataset
from models import MultimodalSentimentModel
from explain import XAIOrchestrator


def main():
    with open('../configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['general']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Використовується пристрій: {device}")

    test_dataset = UrbanSentimentDataset(config['data']['test_csv'], config['data']['image_dir'], config)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MultimodalSentimentModel(config).to(device)
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("Навчена модель успішно завантажена.")
    except FileNotFoundError:
        print("Помилка: файл 'best_model.pth' не знайдено.")
        return

    orchestrator = XAIOrchestrator(model, config, device)

    sample_batch = next(iter(test_loader))

    print("\n--- Запуск XAI Orchestrator для одного семплу ---")
    background_texts = [test_dataset[i]['text'] for i in np.random.choice(len(test_dataset), 10)]

    explanation = orchestrator.explain(sample_batch, background_texts)

    print("\n" + "=" * 50)
    print("          ФІНАЛЬНЕ МУЛЬТИМОДАЛЬНЕ ПОЯСНЕННЯ")
    print("=" * 50)
    print(f"\nПрогноз Моделі: {explanation['prediction']}")
    print("\n--- Текстовий Звіт від LLM Composer ---")
    print(explanation['text_summary'])
    print("\n--- Візуальне Пояснення (Grad-CAM Overlay) ---")

    plt.figure(figsize=(8, 8))
    plt.imshow(explanation['visual_explanation'])
    plt.title("Visual Attribution Map")
    plt.axis('off')
    plt.savefig("final_explanation_visual.png")
    plt.show()
    print("Візуальне пояснення збережено у 'final_explanation_visual.png'")


if __name__ == '__main__':
    main()