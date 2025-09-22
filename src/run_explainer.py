import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import UrbanSentimentDataset
from models import MultimodalSentimentModel
from explain import XAIWrapper


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
        print("Помилка: файл 'best_model.pth' не знайдено. Будь ласка, спочатку запустіть train.py")
        return

    model.eval()

    xai_wrapper = XAIWrapper(model, config, device)

    sample_batch = next(iter(test_loader))
    print("\n--- Обрано семпл для аналізу ---")
    print(f"Текст: {sample_batch['text'][0]}")
    print(f"Справжній лейбл: {config['model']['classes'][sample_batch['label'].item()]}")

    print("\n1. Генерація Grad-CAM...")
    heatmap = xai_wrapper.explain_visual(sample_batch)
    print(f"   Heatmap згенеровано. Розмір: {heatmap.shape}")

    print("\n2. Генерація SHAP значень (це може зайняти хвилину)...")
    background_data = [test_dataset[i]['text'] for i in np.random.choice(len(test_dataset), 10)]

    shap_values = xai_wrapper.explain_textual(sample_batch, background_data)

    print("   SHAP значення згенеровано.")

    print("\n--- Найбільш впливові слова (за SHAP) ---")

    with torch.no_grad():
        logits = model(sample_batch)
        pred_class_idx = torch.argmax(logits, dim=1).item()

    print(f"Аналіз для передбаченого класу: '{config['model']['classes'][pred_class_idx]}'")

    word_attributions = list(zip(shap_values.data[0], shap_values.values[0, :, pred_class_idx]))

    word_attributions.sort(key=lambda x: x[1], reverse=True)

    for word, value in word_attributions[:10]:
        print(f"Слово: '{word}', Внесок: {value:.4f}")


if __name__ == '__main__':
    main()