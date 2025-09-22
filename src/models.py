import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    """Енкодер для текстових даних на основі BERT."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_name = config['model']['text_encoder']['name']

        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        if not config['model']['text_encoder']['trainable']:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, texts):
        """
        Args:
            texts (list of str): Список текстів розміром batch_size.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['preprocessing']['text_max_length']
        )

        inputs = {key: val.to(self.bert.device) for key, val in inputs.items()}

        outputs = self.bert(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding


class ImageEncoder(nn.Module):
    """Енкодер для зображень на основі ResNet."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.base_model.fc = nn.Identity()

        if not config['model']['image_encoder']['trainable']:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Тензор зображень розміром [batch_size, 3, H, W].
        """
        features = self.base_model(images)
        return torch.flatten(features, 1)


class GeoEncoder(nn.Module):
    """
    Спрощений енкодер для гео-даних на основі MLP.
    ПРИМІТКА: Це початкова реалізація. Відповідно до методології,
    вона буде замінена на Graph Neural Network (GNN) на наступних етапах.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        geo_config = config['model']['geo_encoder']

        self.model = nn.Sequential(
            nn.Linear(geo_config['input_dim'], geo_config['hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(geo_config['hidden_dim']),
            nn.Linear(geo_config['hidden_dim'], geo_config['embedding_dim'])
        )

    def forward(self, geo_coords):
        """
        Args:
            geo_coords (torch.Tensor): Тензор координат розміром [batch_size, 2].
        """
        return self.model(geo_coords)


class FusionLayer(nn.Module):
    """
    Шар злиття, що поєднує ембеддинги з різних модальностей.
    Використовує projection для уніфікації розмірів та self-attention для взаємодії.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        fusion_config = config['model']['fusion']

        self.text_projection = nn.Linear(config['model']['text_encoder']['embedding_dim'],
                                         fusion_config['projection_dim'])
        self.image_projection = nn.Linear(config['model']['image_encoder']['embedding_dim'],
                                          fusion_config['projection_dim'])
        self.geo_projection = nn.Linear(config['model']['geo_encoder']['embedding_dim'],
                                        fusion_config['projection_dim'])

        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_config['projection_dim'],
            num_heads=fusion_config['attention_heads'],
            dropout=fusion_config['attention_dropout'],
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(fusion_config['projection_dim'])

    def forward(self, text_emb, image_emb, geo_emb):
        """
        Args:
            text_emb (torch.Tensor): [batch_size, text_dim]
            image_emb (torch.Tensor): [batch_size, image_dim]
            geo_emb (torch.Tensor): [batch_size, geo_dim]
        """

        text_proj = self.text_projection(text_emb)
        image_proj = self.image_projection(image_emb)
        geo_proj = self.geo_projection(geo_emb)

        modalities = torch.stack([text_proj, image_proj, geo_proj], dim=1)

        attn_output, _ = self.attention(modalities, modalities, modalities)

        fused_output = self.layer_norm(modalities + attn_output)

        final_embedding = torch.mean(fused_output, dim=1)

        return final_embedding


class MultimodalSentimentModel(nn.Module):
    """Головна мультимодальна модель для класифікації сентименту."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)
        self.geo_encoder = GeoEncoder(config)
        self.fusion_layer = FusionLayer(config)

        classifier_config = config['model']['classifier']
        self.classifier = nn.Sequential(
            nn.Linear(config['model']['fusion']['projection_dim'], classifier_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(classifier_config['dropout']),
            nn.Linear(classifier_config['hidden_dim'], config['model']['num_classes'])
        )

    def forward(self, batch):
        """
        Args:
            batch (dict): Словник з даними, що містить 'image', 'text', 'geo'.
        """
        text_emb = self.text_encoder(batch['text'])
        image_emb = self.image_encoder(batch['image'])
        geo_emb = self.geo_encoder(batch['geo'])

        fused_emb = self.fusion_layer(text_emb, image_emb, geo_emb)

        logits = self.classifier(fused_emb)

        return logits