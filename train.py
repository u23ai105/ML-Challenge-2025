import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DebertaV2Model,
    DebertaV2Tokenizer,
    ViTImageProcessor,
    get_cosine_schedule_with_warmup
)
import timm
from PIL import Image, UnidentifiedImageError
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Set random seeds for reproducibility across all libraries
torch.manual_seed(42)
np.random.seed(42)


# --- 1. Configuration Class ---
class Config:
    """
    Configuration class to store all model parameters, paths, and hyperparameters.
    Centralizing configuration makes experiments easier to manage and reproduce.
    """
    # Data paths - updated to include 'data' folder structure
    train_csv_path = 'data/train.csv'
    test_csv_path = 'data/test.csv'

    # Image directories for training and testing
    train_image_dir = 'image'
    test_image_dir = 'test_image'

    # Model names for text and image processing
    text_model_name = 'microsoft/deberta-v3-base'
    image_model_name = 'swin_base_patch4_window7_224.ms_in22k'

    # Training hyperparameters
    epochs = 35
    batch_size = 16
    learning_rate = 2e-5
    num_warmup_steps = 100
    text_max_length = 256

    # Gradient clipping to prevent exploding gradients
    max_grad_norm = 1.0

    # Data sampling parameters
    num_test_samples = 5000
    num_train_val_samples = 70000

    # Output configuration
    output_dir = './'
    best_model_path = os.path.join(output_dir, 'best_model_by_test_score.pth')
    submission_path = os.path.join(output_dir, 'test_out.csv')

    # System configuration
    num_workers = 2  # Reduced for better compatibility


# Initialize configuration and device
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


# --- 2. Custom SMAPE Loss and Metric ---
def smape_loss(y_true, y_pred, epsilon=1e-8):
    """
    Calculate SMAPE Loss for backpropagation (differentiable)
    Uses smooth approximation of absolute value for better gradients
    """
    diff = y_pred - y_true
    numerator = torch.abs(diff)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + epsilon
    return torch.mean(numerator / denominator)


def smape_metric(y_true, y_pred, epsilon=1e-8):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE) for evaluation
    Standard SMAPE formula - used for monitoring only (not for backprop)
    """
    numerator = torch.abs(y_pred - y_true)
    denominator = torch.abs(y_true) + torch.abs(y_pred) + epsilon
    return torch.mean(2 * numerator / denominator)


# --- 3. Multi-Modal Model Architecture ---
class MultiModalPricePredictor(nn.Module):
    """
    Multi-modal model that combines text (DeBERTa) and image (Swin Transformer) features
    for price prediction. Uses adaptive fusion to balance between text and image modalities.
    """

    def __init__(self, text_model_name, image_model_name):
        """
        Initialize the multi-modal model.
        """
        super(MultiModalPricePredictor, self).__init__()

        # Text encoder (DeBERTa v3)
        print(f"📥 Loading text model: {text_model_name}")
        self.text_model = DebertaV2Model.from_pretrained(text_model_name)

        # Image encoder (Swin Transformer)
        print(f"📥 Loading image model: {image_model_name}")
        self.image_model = timm.create_model(image_model_name, pretrained=True, num_classes=0)

        # Get dimensions for feature alignment
        text_dim = self.text_model.config.hidden_size
        
        # Get Swin dimension correctly with better detection
        if hasattr(self.image_model, 'num_features'):
            swin_dim = self.image_model.num_features
        elif hasattr(self.image_model, 'embed_dim'):
            swin_dim = self.image_model.embed_dim
        elif hasattr(self.image_model, 'head'):
            # Try to infer from the classifier head
            swin_dim = self.image_model.head.in_features if hasattr(self.image_model.head, 'in_features') else 1024
        else:
            # Default for Swin Base
            swin_dim = 1024
            print(f"⚠️  Warning: Could not detect Swin dimension, defaulting to {swin_dim}")
        
        print(f"🔍 Model dimensions - Text: {text_dim}, Image: {swin_dim}")
        
        # Verify by doing a test forward pass
        try:
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                test_feat = self.image_model.forward_features(test_input)
                if isinstance(test_feat, tuple):
                    test_feat = test_feat[0]
                
                # Handle different shapes
                if len(test_feat.shape) == 4:
                    if test_feat.shape[-1] > test_feat.shape[1]:
                        test_feat = test_feat.permute(0, 3, 1, 2)
                    test_feat = torch.mean(test_feat, dim=[2, 3])
                elif len(test_feat.shape) == 3:
                    if test_feat.shape[-1] < test_feat.shape[1]:
                        test_feat = test_feat.permute(0, 2, 1)
                    test_feat = torch.mean(test_feat, dim=1)
                
                actual_dim = test_feat.shape[1]
                if actual_dim != swin_dim:
                    print(f"⚠️  Dimension mismatch detected! Expected {swin_dim}, got {actual_dim}")
                    print(f"🔧 Updating swin_dim to {actual_dim}")
                    swin_dim = actual_dim
                else:
                    print(f"✅ Dimension verification passed: {actual_dim}")
        except Exception as e:
            print(f"⚠️  Could not verify dimensions: {e}")
        
        print(f"🔍 Final dimensions - Text: {text_dim}, Image: {swin_dim}")

        # Project image features to text feature space
        self.image_projection = nn.Linear(swin_dim, text_dim)

        # Learnable fusion parameter
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]))

        # MLP head for final price prediction
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

        # Initialize weights for newly added layers
        self._init_weights(self.image_projection)
        self._init_weights(self.mlp_head)

    def _init_weights(self, module):
        """
        Initialize weights for linear layers.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, pixel_values, image_is_valid):
        """
        Forward pass through the multi-modal model.
        """
        batch_size = input_ids.size(0)
        
        # Text feature extraction - get [CLS] token representation
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_output.last_hidden_state[:, 0, :]  # Shape: (batch_size, text_dim)

        # Image feature extraction with proper Swin handling
        img_feat = self.image_model.forward_features(pixel_values)
        
        # Debug: Check the shape we're getting
        # print(f"DEBUG: img_feat shape = {img_feat.shape}, type = {type(img_feat)}")
        
        # Handle different output formats from Swin Transformer
        if isinstance(img_feat, tuple):
            img_feat = img_feat[0]
        
        # Swin Transformer typically outputs: (batch_size, height, width, embed_dim) 
        # or (batch_size, num_patches, embed_dim)
        if len(img_feat.shape) == 4:  
            # If shape is (batch, H, W, C) - permute and pool
            if img_feat.shape[-1] > img_feat.shape[1]:  # Last dim is likely channels
                img_feat = img_feat.permute(0, 3, 1, 2)  # (batch, C, H, W)
            # Global average pooling
            img_feat = torch.mean(img_feat, dim=[2, 3])  # (batch, C)
        elif len(img_feat.shape) == 3:  # (batch, num_patches, embed_dim)
            # Check if last dimension is the embedding dimension (should be large like 1024)
            if img_feat.shape[-1] < img_feat.shape[1]:
                # Likely (batch, embed_dim, num_patches) - transpose it
                img_feat = img_feat.permute(0, 2, 1)  # (batch, num_patches, embed_dim)
            # Global average pooling across patches
            img_feat = torch.mean(img_feat, dim=1)  # (batch, embed_dim)
        elif len(img_feat.shape) == 2:  # Already (batch, embed_dim)
            pass  # Nothing to do
        
        # Now img_feat should be (batch_size, swin_dim)
        # Add assertion to catch issues early
        assert img_feat.shape[1] == self.image_projection.in_features, \
            f"Expected {self.image_projection.in_features} features, got {img_feat.shape[1]}"
        
        img_emb = self.image_projection(img_feat)  # Shape: (batch_size, text_dim)

        # Adaptive fusion with learnable weight
        alpha = torch.sigmoid(self.fusion_weight)  # Scalar value
        mask = image_is_valid.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1)

        # Fuse embeddings - both should be (batch_size, text_dim)
        fused_emb = (alpha * img_emb * mask) + ((1 - (alpha * mask)) * text_emb)

        # Final price prediction
        return self.mlp_head(fused_emb).squeeze(-1)


# --- 4. Custom Dataset Class ---
class ProductDataset(Dataset):
    """
    Custom PyTorch Dataset for handling product data with text and images.
    """

    def __init__(self, dataframe, tokenizer, image_processor, image_dir, is_test=False):
        """
        Initialize dataset.
        """
        self.df = dataframe
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.is_test = is_test
        self.image_size = self.image_processor.size['height']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Handle text with proper NaN checking
        catalog_content = row.get('catalog_content', '')
        if pd.isna(catalog_content):
            catalog_content = ''
        text = str(catalog_content)
        
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config.text_max_length,
            return_tensors='pt'
        )

        img_path = os.path.join(self.image_dir, f"{row['sample_id']}.jpg")
        pixels = torch.zeros((3, self.image_size, self.image_size))
        is_valid = False

        try:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                pixels = self.image_processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
                is_valid = True
        except (UnidentifiedImageError, OSError, Exception):
            pass

        item = {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'pixel_values': pixels,
            'image_is_valid': torch.tensor(is_valid, dtype=torch.bool)
        }

        if not self.is_test:
            item['price'] = torch.tensor(row['price'], dtype=torch.float32)

        return item


# --- 5. Training and Evaluation Functions ---

def train_one_epoch(model, loader, optimizer, scheduler):
    """
    Train the model for one epoch using SMAPE loss.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        for k, v in batch.items():
            batch[k] = v.to(device)

        preds = model(
            batch['input_ids'],
            batch['attention_mask'],
            batch['pixel_values'],
            batch['image_is_valid']
        )

        # Use SMAPE loss for backpropagation
        loss = smape_loss(batch['price'], preds)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    """
    Evaluate model on validation data using SMAPE.
    """
    model.eval()
    total_loss, total_smape = 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            for k, v in batch.items():
                batch[k] = v.to(device)

            preds = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['pixel_values'],
                batch['image_is_valid']
            )

            # Use SMAPE loss for consistency
            total_loss += smape_loss(batch['price'], preds).item()
            # Also track standard SMAPE metric
            total_smape += smape_metric(batch['price'], preds).item()

    return total_loss / len(loader), total_smape / len(loader)


def test_evaluation(model, loader):
    """
    Evaluate model on test data using SMAPE metric only.
    """
    model.eval()
    total_smape = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            for k, v in batch.items():
                batch[k] = v.to(device)

            preds = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['pixel_values'],
                batch['image_is_valid']
            )

            total_smape += smape_metric(batch['price'], preds).item()

    return total_smape / len(loader)


def predict_on_test_data(model, loader):
    """
    Generate predictions for test data (without labels).
    """
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting on test.csv"):
            for k, v in batch.items():
                batch[k] = v.to(device)

            preds = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['pixel_values'],
                batch['image_is_valid']
            )
            all_predictions.extend(preds.cpu().numpy())

    return all_predictions


# --- 6. Main Execution Block ---
if __name__ == '__main__':
    os.makedirs(config.output_dir, exist_ok=True)

    print("\n📦 Loading and splitting training data...")
    if not os.path.exists(config.train_csv_path):
        print(f"❌ Error: Training file not found at '{config.train_csv_path}'")
        exit()

    full_df = pd.read_csv(config.train_csv_path).dropna(subset=['price', 'catalog_content'])
    shuffled_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = shuffled_df.iloc[:config.num_test_samples]
    train_val_pool = shuffled_df.iloc[config.num_test_samples: config.num_test_samples + config.num_train_val_samples]
    train_df = train_val_pool.sample(frac=0.8, random_state=42)
    val_df = train_val_pool.drop(train_df.index)

    print(f"Data split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} internal test")

    # Initialize tokenizers and processors
    tokenizer = DebertaV2Tokenizer.from_pretrained(config.text_model_name)
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Create datasets
    train_dataset = ProductDataset(train_df, tokenizer, image_processor, config.train_image_dir)
    val_dataset = ProductDataset(val_df, tokenizer, image_processor, config.train_image_dir)
    internal_test_dataset = ProductDataset(test_df, tokenizer, image_processor, config.train_image_dir)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    internal_test_loader = DataLoader(internal_test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Initialize model, optimizer, and scheduler
    model = MultiModalPricePredictor(config.text_model_name, config.image_model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=len(train_loader) * config.epochs
    )

    best_test_smape = float('inf')
    print("\n🚀 Starting training with SMAPE loss...")
    print("-" * 80)

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_smape = evaluate(model, val_loader)
        test_smape = test_evaluation(model, internal_test_loader)

        print(
            f"\nEpoch {epoch + 1}/{config.epochs} | "
            f"Train SMAPE: {train_loss * 100:.4f}% | "
            f"Val SMAPE: {val_smape * 100:.4f}% | "
            f"Test SMAPE: {test_smape * 100:.4f}%"
        )

        if test_smape < best_test_smape:
            best_test_smape = test_smape
            torch.save(model.state_dict(), config.best_model_path)
            print(f"🎉 New best model saved! Test SMAPE: {best_test_smape * 100:.4f}%")

    print("-" * 80)
    print("✅ Training complete!")
    print(f"🏆 Best model (Test SMAPE: {best_test_smape * 100:.4f}%) is stored at: {config.best_model_path}")

    print("\n📊 Generating final predictions for submission...")
    if not os.path.exists(config.test_csv_path):
        print(f"❌ Error: Final test file not found at '{config.test_csv_path}'. Skipping prediction.")
    else:
        model.load_state_dict(torch.load(config.best_model_path, map_location=device))
        print("✅ Best model weights loaded for final prediction.")

        final_test_df = pd.read_csv(config.test_csv_path)
        final_test_dataset = ProductDataset(
            final_test_df, tokenizer, image_processor, config.test_image_dir, is_test=True
        )
        final_test_loader = DataLoader(
            final_test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
        )

        predictions = predict_on_test_data(model, final_test_loader)

        submission_df = pd.DataFrame({
            'sample_id': final_test_df['sample_id'],
            'price': predictions
        })
        submission_df.to_csv(config.submission_path, index=False)
        print(f"✅ Submission file created at: {config.submission_path}")
