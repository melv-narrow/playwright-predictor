"""
Prepare training data for the element classifier model.
"""

import logging
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Dict, List
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class DataPreparator:
    def __init__(self, data_dir: str = 'data/normalized'):
        self.data_dir = Path(data_dir)
        self.normalized_dir = self.data_dir
        self.processed_dir = Path('training_data/processed')
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        # Initialize label encoders
        self.element_type_encoder = LabelEncoder()
        self.semantic_role_encoder = LabelEncoder()
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'data_preparation.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_normalized_data(self) -> pd.DataFrame:
        """Load and combine all normalized datasets."""
        dfs = []
        for file in self.normalized_dir.glob('*_normalized.csv'):
            try:
                df = pd.read_csv(file)
                df['source'] = file.stem.replace('_normalized', '')
                dfs.append(df)
                logging.info(f"Loaded {len(df)} records from {file.name}")
            except Exception as e:
                logging.error(f"Error loading {file}: {str(e)}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Combined dataset size: {len(combined_df)} records")
        return combined_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare features and labels for training."""
        # Encode labels
        element_types = self.element_type_encoder.fit_transform(df['element_type'])
        semantic_roles = self.semantic_role_encoder.fit_transform(df['semantic_role'])
        
        # Save label encoders
        with open(self.processed_dir / 'element_type_encoder.json', 'w') as f:
            json.dump({
                'classes_': self.element_type_encoder.classes_.tolist()
            }, f)
        
        with open(self.processed_dir / 'semantic_role_encoder.json', 'w') as f:
            json.dump({
                'classes_': self.semantic_role_encoder.classes_.tolist()
            }, f)
        
        # Prepare text features
        text_features = []
        for _, row in df.iterrows():
            # Combine relevant text fields
            text = f"""
            Tag: {row['normalized_tag']}
            Text: {row['inner_text']}
            Selector: {row['selector']}
            Attributes: {row['attributes']}
            """
            text_features.append(text)
        
        # Tokenize text
        encodings = self.tokenizer(
            text_features,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Convert boolean columns to float
        bool_columns = ['is_interactive', 'is_visible', 'has_text']
        for col in bool_columns:
            df[col] = df[col].astype(float)
        
        # Prepare numerical features
        numerical_features = torch.tensor(df[[
            'is_interactive',
            'is_visible',
            'has_text'
        ]].values, dtype=torch.float32)
        
        # Combine features
        features = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'numerical_features': numerical_features
        }
        
        # Prepare multi-task labels
        labels = torch.tensor(np.column_stack([element_types, semantic_roles]), dtype=torch.long)
        
        return features, labels
    
    def prepare_data_loaders(
        self, 
        features: Dict[str, torch.Tensor], 
        labels: torch.Tensor,
        batch_size: int = 32,
        val_size: float = 0.2,
        test_size: float = 0.1
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        # First split into train and temp
        train_features = {k: v.clone() for k, v in features.items()}
        temp_features = {k: v.clone() for k, v in features.items()}
        
        train_idx, temp_idx = train_test_split(
            range(len(labels)), 
            test_size=(val_size + test_size),
            random_state=42
        )
        
        for k in train_features:
            train_features[k] = features[k][train_idx]
            temp_features[k] = features[k][temp_idx]
        
        train_labels = labels[train_idx]
        temp_labels = labels[temp_idx]
        
        # Then split temp into val and test
        val_features = {k: v.clone() for k, v in temp_features.items()}
        test_features = {k: v.clone() for k, v in temp_features.items()}
        
        val_size = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            range(len(temp_labels)),
            test_size=(1 - val_size),
            random_state=42
        )
        
        for k in val_features:
            val_features[k] = temp_features[k][val_idx]
            test_features[k] = temp_features[k][test_idx]
        
        val_labels = temp_labels[val_idx]
        test_labels = temp_labels[test_idx]
        
        # Create datasets
        train_dataset = ElementDataset(train_features, train_labels)
        val_dataset = ElementDataset(val_features, val_labels)
        test_dataset = ElementDataset(test_features, test_labels)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        logging.info(f"Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def prepare_data(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Main method to prepare training data."""
        # Load data
        df = self.load_normalized_data()
        
        # Prepare features and labels
        features, labels = self.prepare_features(df)
        
        # Create data loaders
        return self.prepare_data_loaders(features, labels, batch_size)

class ElementDataset(Dataset):
    def __init__(self, features: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.features.items()}, self.labels[idx]

def main():
    preparer = DataPreparator()
    train_loader, val_loader, test_loader = preparer.prepare_data()
    
    # Save some statistics
    stats = {
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
        'element_types': preparer.element_type_encoder.classes_.tolist(),
        'semantic_roles': preparer.semantic_role_encoder.classes_.tolist(),
    }
    
    with open(preparer.processed_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info("Data preparation complete!")

if __name__ == "__main__":
    main()
