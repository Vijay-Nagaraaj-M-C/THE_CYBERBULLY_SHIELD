import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import DistilBertModel

class CyberbullyShieldFusion(nn.Module):
    """
    Multimodal framework combining DistilBERT (text) and ResNet50 (images)
    using Cross-Attention, tailored for 4GB VRAM.
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super(CyberbullyShieldFusion, self).__init__()
        
        # 1. Textual Analysis Branch (DistilBERT)
        # Using the base uncased version to get (batch, seq_len, 768)
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # 2. Visual Analysis Branch (ResNet50)
        # Loading ResNet50 and dropping the AvgPool & FC layers.
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # We want spatial feature maps: output of layer4 is (batch, 2048, 7, 7)
        modules = list(resnet.children())[:-2] 
        self.vision_model = nn.Sequential(*modules)
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        # Projection of vision features to match text embed_dim
        # ResNet50 layer4 output channels = 2048
        self.vision_proj = nn.Linear(2048, embed_dim)
        
        # 3. Fusion Engine (Cross-Attention)
        # Queries: Image embeddings
        # Keys/Values: Text embeddings
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Final Classifier
        # Feed forward network outputting binary probability logit
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1) # Single output for BCEWithLogitsLoss
        )

    def unfreeze_top_layers(self):
        """Unfreeze the top layers for fine-tuning Phase 2."""
        for name, param in self.text_model.named_parameters():
            if 'transformer.layer.5' in name:
                param.requires_grad = True
                
        for name, param in self.vision_model.named_parameters():
            if '7.' in name:
                param.requires_grad = True

    def train(self, mode=True):
        """Keep frozen baseline models in eval mode to prevent batchnorm issues."""
        super(CyberbullyShieldFusion, self).train(mode)
        if mode:
            self.text_model.eval()
            self.vision_model.eval()
        return self

    def forward(self, input_ids, attention_mask, image_tensor):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            image_tensor: (batch_size, 3, 224, 224)
        """
        batch_size = input_ids.size(0)
        
        # --- TEXT FORWARD ---
        # Outputs: (batch_size, seq_len, 768)
        text_outputs = self.text_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        # We need the full sequence embeddings for attention keys/values
        text_features = text_outputs.last_hidden_state 
        
        # --- VISION FORWARD ---
        # Outputs: (batch_size, 2048, 7, 7)
        vision_outputs = self.vision_model(image_tensor)
        
        # Flatten spatial dimensions: (batch, 2048, 49)
        vision_features = vision_outputs.view(batch_size, 2048, -1)
        # Permute to (batch, 49, 2048) so spatial spots act as sequence elements
        vision_features = vision_features.permute(0, 2, 1)
        
        # Project visual embeddings to text dimensions (batch, 49, 768)
        vision_proj = self.vision_proj(vision_features)
        
        # --- CROSS-ATTENTION FUSION ---
        # Queries (Q): Vision  (batch, 49, 768)
        # Keys (K), Values (V): Text (batch, seq_len, 768)
        # We need attention mask inverted for MultiheadAttention if we want to mask PAD tokens
        # PyTorch MHA expected key_padding_mask: True means ignore exactly those keys.
        # But distilbert attention_mask has 0 for pad. So we invert it.
        key_padding_mask = (attention_mask == 0)
        
        attn_output, _ = self.cross_attention(
            query=vision_proj, 
            key=text_features, 
            value=text_features, 
            key_padding_mask=key_padding_mask
        )
        # Output shape is (batch, 49, 768)
        
        # --- CLASSIFIER ---
        # Pool the cross-attended features (e.g. mean across spatial dimension)
        # (batch, 768)
        pooled_output = attn_output.mean(dim=1)
        
        # Output logits (batch, 1)
        logits = self.classifier(pooled_output)
        
        # Squeeze the trailing dimension if necessary
        return logits.squeeze(-1)
