import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from dataset import UrbanSentimentDataset
from models import MultimodalSentimentModel
from explain import XAIOrchestrator, GradCAM, apply_gradcam_overlay


def generate_full_gradcam_figure(orchestrator, sample_batch, filename):
    """
    Генерує та зберігає повну 4-панельну візуалізацію Grad-CAM,
    як у вашій статті (Figure 7, 8).
    """
    model = orchestrator.model
    device = orchestrator.device

    heatmap, pred_class_idx = orchestrator.xai_wrapper.explain_visual(sample_batch)

    original_image_tensor = sample_batch['original_image_tensor'][0]
    original_image_np = (original_image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    overlay_img = apply_gradcam_overlay(heatmap, original_image_np)

    heatmap_resized = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
    mask = heatmap_resized > 0.5
    gray_image = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2GRAY)
    gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    focus_mask_img = np.where(mask[..., np.newaxis], original_image_np, gray_image_3ch)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    prediction_class = orchestrator.config['model']['classes'][pred_class_idx]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Full Analysis using Grad-CAM for '{prediction_class}' class", fontsize=16)

    axes[0, 0].imshow(original_image_np)
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(heatmap_colored)
    axes[0, 1].set_title("2. Grad-CAM Heatmap")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(overlay_img)
    axes[1, 0].set_title(f"3. Overlay Map\nPrediction: {prediction_class}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(focus_mask_img)
    axes[1, 1].set_title("4. Focus Areas (Mask)")
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300)
    print(f"Фігуру збережено у файл: {filename}")
    plt.close(fig)


def main():
    with open('../configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    print(f"Використовується пристрій: {device}")

    test_dataset = UrbanSentimentDataset(config['data']['test_csv'], config['data']['image_dir'], config)

    model = MultimodalSentimentModel(config).to(device)
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("Навчена модель успішно завантажена.")
    except FileNotFoundError:
        print("Помилка: файл 'best_model.pth' не знайдено.")
        return

    orchestrator = XAIOrchestrator(model, config, device)

    SAMPLE_IDS_TO_VISUALIZE = {
        'figure_7': 're_00001',
        'figure_9_failure_case': 're_00002'
    }

    sample_indices = {
        name: test_dataset.metadata[test_dataset.metadata['sample_id'] == sample_id].index[0]
        for name, sample_id in SAMPLE_IDS_TO_VISUALIZE.items()
    }

    print("\n--- Генерація Figure 7: Urban Decay Case ---")
    fig7_dataset = Subset(test_dataset, [sample_indices['figure_7']])
    fig7_loader = DataLoader(fig7_dataset, batch_size=1)
    fig7_batch = next(iter(fig7_loader))
    generate_full_gradcam_figure(orchestrator, fig7_batch, "Figure_7_Urban_Decay.png")

    print("\n--- Генерація Figure 9: Failure Case ---")
    fig9_dataset = Subset(test_dataset, [sample_indices['figure_9_failure_case']])
    fig9_loader = DataLoader(fig9_dataset, batch_size=1)
    fig9_batch = next(iter(fig9_loader))
    generate_full_gradcam_figure(orchestrator, fig9_batch, "Figure_9_Failure_Case.png")

    print("\nГенерація фігур для статті завершена.")


if __name__ == '__main__':
    main()