import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import DistilBertTokenizer
import easyocr
import pandas as pd
from sklearn.model_selection import train_test_split
import glob

# Initialize the EasyOCR reader (this will run on GPU if available)
# Keep it global to avoid re-initializing per image
OCR_READER = easyocr.Reader(['en'], gpu=True)

def clean_text(text):
    """
    Cleans text STRICTLY using the `re` module (NO NLTK).
    - Removes URLs
    - Removes HTML tags
    - Removes excess whitespace
    - Strips # and @ symbols while keeping the attached words.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Strip @ and # keeping the attached words (e.g., @user -> user, #bullying -> bullying)
    text = re.sub(r'[@#]([A-Za-z0-9_]+)', r'\1', text)
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_from_image(image_path):
    """
    Uses EasyOCR to extract text directly from the image.
    Returns the extracted text as a single string.
    """
    if not os.path.exists(image_path):
        return ""
    try:
        results = OCR_READER.readtext(image_path)
        extracted_text = " ".join([res[1] for res in results])
        return extracted_text
    except Exception as e:
        print(f"Warning: OCR failed for {image_path}: {e}")
        return ""

class UnifiedCyberbullyDataset(Dataset):
    """
    A unified PyTorch Dataset handling the 3 specific datasets:
    1. saurabhshahane/cyberbullying-dataset (Text-only)
    2. soorajtomar/cyberbullying-tweets (Text-only)
    3. studentramya/multimodal-cyberbullying (Text + Images)
    
    This class handles the merging, missing modality replacements (zero-tensors, empty strings),
    and normalizes the outputs.
    """
    def __init__(self, dataframe, tokenizer, max_length=128):
        """
        Args:
            dataframe: A Pandas DataFrame containing ['text', 'image_path', 'label'] columns.
                       If a modality is missing, the corresponding entry should be None or NaN.
            tokenizer: HuggingFace DistilBertTokenizer
            max_length: max length for text tokenization
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        raw_text = row['text'] if pd.notna(row['text']) else ""
        image_path = row['image_path'] if pd.notna(row['image_path']) else None
        label = float(row['label']) 
        
        # 1. Process Image Branch
        image_tensor = None
        extracted_text = ""
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                image_tensor = self.transform(img)
                # Extract text using EasyOCR and append it to our textual branch
                extracted_text = extract_text_from_image(image_path)
            except Exception as e:
                # If image loading fails, fallback to zero-tensor
                image_tensor = torch.zeros((3, 224, 224))
        else:
            # Missing visual modality: generate zero-tensor
            image_tensor = torch.zeros((3, 224, 224))

        # 2. Process Text Branch
        combined_text = f"{raw_text} {extracted_text}".strip()
        cleaned_text = clean_text(combined_text)
        
        # Missing textual modality fallback
        if not cleaned_text:
            cleaned_text = "" # empty string as requested

        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image_tensor': image_tensor,
            'label': torch.tensor(label, dtype=torch.float)
        }

def get_dataloaders(train_df, val_df, batch_size=4):
    """
    Creates DataLoaders for train and validation data.
    Batch size 4 or 8 is enforced for the 4GB VRAM constraint.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_dataset = UnifiedCyberbullyDataset(train_df, tokenizer)
    val_dataset = UnifiedCyberbullyDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader

def load_all_datasets(datasets_dir=None):
    """
    Loads all locally available datasets from the 3 specified folders into a single DataFrame.
    Returns: train_df, val_df
    """
    if datasets_dir is None:
        datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets'))
    
    all_data = []

    # 1. soorajtomar/cyberbullying-tweets (Text-only)
    tweets_path = os.path.join(datasets_dir, "cyberbullying-tweets.csv")
    if os.path.exists(tweets_path):
        df_tweets = pd.read_csv(tweets_path)
        # Assuming Columns: 'Text', 'CB_Label'
        if 'Text' in df_tweets.columns and 'CB_Label' in df_tweets.columns:
            df = pd.DataFrame({
                'text': df_tweets['Text'].astype(str),
                'image_path': None,
                'label': df_tweets['CB_Label'].astype(float)
            })
            all_data.append(df)
            print(f"Loaded {len(df)} rows from cyberbullying-tweets.csv")

    # 2. saurabhshahane/cyberbullying-dataset (Text-only)
    cb_dataset_dir = os.path.join(datasets_dir, "cyberbullying-dataset")
    if os.path.exists(cb_dataset_dir):
        for csv_file in glob.glob(os.path.join(cb_dataset_dir, "*.csv")):
            df_cb = pd.read_csv(csv_file)
            # Assuming Columns: 'Text', 'oh_label'
            if 'Text' in df_cb.columns and 'oh_label' in df_cb.columns:
                df = pd.DataFrame({
                    'text': df_cb['Text'].astype(str),
                    'image_path': None,
                    'label': df_cb['oh_label'].astype(float) # usually 0/1
                })
                all_data.append(df)
        print(f"Loaded text files from {cb_dataset_dir}")

    # 3. studentramya/multimodal-cyberbullying (Text + Images)
    multimodal_csv = os.path.join(datasets_dir, "multimodal-cyberbullying", "cyberbully.csv")
    img_dir = os.path.join(datasets_dir, "multimodal-cyberbullying", "bully_data")
    if os.path.exists(multimodal_csv):
        df_mm = pd.read_csv(multimodal_csv)
        # Assuming Columns: 'Img-Name', 'Img-Text', 'Img-Label'
        if all(c in df_mm.columns for c in ['Img-Name', 'Img-Text', 'Img-Label']):
            df_mm['label'] = df_mm['Img-Label'].apply(lambda x: 1.0 if str(x).lower().strip() == 'bully' else 0.0)
            df = pd.DataFrame({
                'text': df_mm['Img-Text'].astype(str),
                'image_path': df_mm['Img-Name'].apply(lambda n: os.path.join(img_dir, str(n))),
                'label': df_mm['label']
            })
            all_data.append(df)
            print(f"Loaded {len(df)} rows from multimodal-cyberbullying")

    if not all_data:
        raise ValueError(f"No valid data found in {datasets_dir}. Please check your dataset headers.")

    final_df = pd.concat(all_data, ignore_index=True)
    
    # Drop rows where label is NaN to prevent train_test_split from throwing ValueError
    final_df = final_df.dropna(subset=['label'])
    
    # Shuffle and Split 90/10 Train/Val
    train_df, val_df = train_test_split(final_df, test_size=0.1, random_state=42, stratify=final_df['label'])
    
    print(f"Total Unified Samples: {len(final_df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df, val_df
