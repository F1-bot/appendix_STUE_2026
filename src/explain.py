import torch
import torch.nn.functional as F
import numpy as np
import cv2
import shap
import requests
import json
from transformers import BertTokenizer


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model;
        self.target_layer = target_layer
        self.gradients = None;
        self.activations = None;
        self.hook_handles = []
        self.hook_handles.append(self.target_layer.register_forward_hook(self.save_activation))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, m, i, o):
        self.activations = o

    def save_gradient(self, m, gi, go):
        self.gradients = go[0]

    def generate(self, model_input_batch, class_idx=None):
        self.model.eval()

        original_grad_state = {p: p.requires_grad for p in self.model.parameters()}

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.zero_grad()

        model_output = self.model(model_input_batch)

        if class_idx is None:
            class_idx = torch.argmax(model_output, dim=1).item()

        class_score = model_output[:, class_idx]
        class_score.backward()

        if self.gradients is None or self.activations is None:
            for p, state in original_grad_state.items():
                p.requires_grad = state
            raise RuntimeError("Could not retrieve gradients or activations.")

        gradients = self.gradients.cpu()
        activations = self.activations.cpu()

        for p, state in original_grad_state.items():
            p.requires_grad = state

        weights = torch.mean(gradients, dim=[2, 3])
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().numpy(), class_idx

    def remove_hooks(self):
        for handle in self.hook_handles: handle.remove()


def apply_gradcam_overlay(heatmap, original_image, alpha=0.5):
    """Накладає heatmap на оригінальне зображення."""
    if isinstance(original_image, torch.Tensor):
        img = original_image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        original_image = (img * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), 1 - alpha, heatmap_colored,
                                       alpha, 0)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)


class XAIWrapper:
    def __init__(self, model, config, device):
        self.model = model;
        self.config = config;
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(config['model']['text_encoder']['name'])
        self.model.eval()

    def explain_visual(self, sample_batch):
        """Генерує Grad-CAM для зображення."""

        def get_nested_attr(obj, attr_string):
            """Допоміжна функція для отримання вкладеного атрибута за рядком."""
            attrs = attr_string.split('.')
            for attr in attrs:
                obj = getattr(obj, attr)
            return obj

        try:
            target_layer_path = self.config['xai']['grad_cam_target_layer']
            target_layer = get_nested_attr(self.model, target_layer_path)
        except AttributeError as e:
            print(f"Помилка: не вдалося знайти цільовий шар за шляхом '{target_layer_path}'. Перевірте конфігурацію.")
            print(f"Деталі помилки: {e}")
            return None, None

        grad_cam = GradCAM(self.model, target_layer)

        batch_for_device = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in sample_batch.items()}

        heatmap, class_idx = grad_cam.generate(batch_for_device)
        grad_cam.remove_hooks()

        return heatmap, class_idx

    def explain_textual(self, sample_batch, background_texts):
        def predict_for_shap(texts):
            texts = list(texts)
            batch_size = len(texts)
            batch_for_device = {
                'image': sample_batch['image'].repeat(batch_size, 1, 1, 1).to(self.device),
                'text': texts,
                'geo': sample_batch['geo'].repeat(batch_size, 1).to(self.device)
            }
            with torch.no_grad():
                logits = self.model(batch_for_device)
                probabilities = F.softmax(logits, dim=1)
            return probabilities.cpu().numpy()

        explainer = shap.Explainer(predict_for_shap, shap.maskers.Text(self.tokenizer))
        shap_values = explainer(sample_batch['text'], fixed_context=1)
        return shap_values


class XAIOrchestrator:
    """Оркестратор, що поєднує все в єдину систему."""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.xai_wrapper = XAIWrapper(model, config, device)

    def _prepare_prompt(self, prediction_class, confidence, shap_values, pred_class_idx):
        """Готує фінальний промпт для LLM."""

        word_attributions = sorted(
            list(zip(shap_values.data[0], shap_values.values[0, :, pred_class_idx])),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]

        evidence = {
            "prediction": {"class": prediction_class, "confidence": f"{confidence:.2f}"},
            "visual_evidence": {
                "source": "Grad-CAM",
                "description": "The model's visual attention was focused on specific regions of the image, indicating their high importance for the prediction."
            },
            "textual_evidence": {
                "source": "SHAP",
                "top_features": [{"feature": word, "importance_score": f"{score:.4f}"} for word, score in
                                 word_attributions]
            }
        }

        prompt = f"""
SYSTEM PROMPT: You are an expert AI analyst. Your task is to synthesize the provided structured evidence into a concise, objective explanation for an urban planner. You must only use the information given below. Do not add any information not present in the evidence. Structure your response into two paragraphs: the first summarizing the prediction and its primary drivers, the second connecting the evidence from different modalities. Do not reveal your identity as an AI.

INPUT DATA (JSON Format):
{json.dumps(evidence, indent=2)}

OUTPUT FORMAT AND INSTRUCTIONS:
## Overall Assessment
[State the final sentiment prediction and its confidence level. Identify the single most influential feature across all modalities that was the primary driver for this prediction.]

## Evidence Synthesis
[In this paragraph, connect the evidence from the different modalities. Explain how the visual evidence corroborates or complements the textual evidence.]
"""
        return prompt

    def _query_llm(self, prompt):
        """Надсилає запит до локального LLM через Ollama."""
        llm_config = self.config['xai']['llm_composer']
        try:
            response = requests.post(
                llm_config['api_endpoint'],
                json={"model": llm_config['model_name'], "prompt": prompt, "stream": False},
                timeout=60
            )
            response.raise_for_status()
            return response.json().get('response', "Error: Could not parse LLM response.")
        except requests.exceptions.RequestException as e:
            return f"Error: Could not connect to Ollama at {llm_config['api_endpoint']}. Please ensure it is running. Details: {e}"

    def explain(self, sample_batch, background_texts):
        """
        Головний метод, що генерує повне мультимодальне пояснення.
        """
        print("1. Running prediction...")
        with torch.no_grad():
            logits = self.model(sample_batch)
            probabilities = F.softmax(logits, dim=1)
            confidence, pred_class_idx = torch.max(probabilities, dim=1)
            pred_class_idx = pred_class_idx.item()
            confidence = confidence.item()
            prediction_class = self.config['model']['classes'][pred_class_idx]

        print(f"   -> Prediction: '{prediction_class}' with confidence {confidence:.2f}")

        print("2. Generating raw explanations (Grad-CAM & SHAP)...")
        heatmap, _ = self.xai_wrapper.explain_visual(sample_batch)
        shap_values = self.xai_wrapper.explain_textual(sample_batch, background_texts)

        print("3. Preparing prompt and querying LLM...")
        prompt = self._prepare_prompt(prediction_class, confidence, shap_values, pred_class_idx)
        text_summary = self._query_llm(prompt)

        print("4. Composing final visual explanation...")
        visual_explanation = apply_gradcam_overlay(heatmap, sample_batch['image'][0])

        return {
            'visual_explanation': visual_explanation,
            'text_summary': text_summary,
            'prediction': prediction_class
        }
