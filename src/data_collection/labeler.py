"""
Data Labeler

Interactive tool for labeling collected training data and validating automated labels.
"""

import pandas as pd
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
from loguru import logger
from bs4 import BeautifulSoup
import webbrowser

class DataLabeler:
    def __init__(self, dataset_path: str):
        """
        Initialize the data labeler.
        
        Args:
            dataset_path: Path to collected dataset CSV
        """
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.current_index = 0
        self.action_types = ['click', 'input', 'select', 'check', 'assert', 'ignore']
        
        # Initialize UI
        self.root = tk.Tk()
        self.root.title("AI Test Generator - Data Labeler")
        self.root.geometry("1200x800")
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the labeling interface."""
        # Element preview
        preview_frame = ttk.LabelFrame(self.root, text="Element Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.html_preview = tk.Text(preview_frame, height=10, wrap=tk.WORD)
        self.html_preview.pack(fill=tk.BOTH, expand=True)
        
        # Element properties
        props_frame = ttk.LabelFrame(self.root, text="Element Properties", padding=10)
        props_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.props_text = tk.Text(props_frame, height=8, wrap=tk.WORD)
        self.props_text.pack(fill=tk.BOTH, expand=True)
        
        # Labeling controls
        controls_frame = ttk.Frame(self.root, padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Action selection
        ttk.Label(controls_frame, text="Action:").pack(side=tk.LEFT, padx=5)
        self.action_var = tk.StringVar()
        action_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.action_var,
            values=self.action_types
        )
        action_combo.pack(side=tk.LEFT, padx=5)
        
        # Navigation buttons
        ttk.Button(
            controls_frame,
            text="Previous",
            command=self._prev_element
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            controls_frame,
            text="Next",
            command=self._next_element
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            controls_frame,
            text="View in Browser",
            command=self._view_in_browser
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            controls_frame,
            text="Save",
            command=self._save_labels
        ).pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress_var = tk.StringVar()
        ttk.Label(
            controls_frame,
            textvariable=self.progress_var
        ).pack(side=tk.RIGHT, padx=5)
        
        # Load first element
        self._load_current_element()
        
    def _load_current_element(self):
        """Load current element data into the UI."""
        if self.current_index >= len(self.df):
            return
            
        row = self.df.iloc[self.current_index]
        
        # Update HTML preview
        self.html_preview.delete('1.0', tk.END)
        self.html_preview.insert('1.0', row['html'])
        
        # Update properties
        props = {
            'URL': row['url'],
            'Tag': row['tag_name'],
            'Text': row['inner_text'],
            'Visible': row['is_visible'],
            'Suggested Action': row['suggested_action']
        }
        
        self.props_text.delete('1.0', tk.END)
        self.props_text.insert('1.0', json.dumps(props, indent=2))
        
        # Update action
        self.action_var.set(row['suggested_action'])
        
        # Update progress
        self.progress_var.set(
            f"Element {self.current_index + 1} of {len(self.df)}"
        )
        
    def _next_element(self):
        """Move to next element."""
        self._save_current_label()
        self.current_index = min(self.current_index + 1, len(self.df) - 1)
        self._load_current_element()
        
    def _prev_element(self):
        """Move to previous element."""
        self._save_current_label()
        self.current_index = max(self.current_index - 1, 0)
        self._load_current_element()
        
    def _save_current_label(self):
        """Save label for current element."""
        if self.current_index < len(self.df):
            self.df.at[self.current_index, 'verified_action'] = self.action_var.get()
            
    def _view_in_browser(self):
        """Open current element's page in browser."""
        if self.current_index < len(self.df):
            url = self.df.iloc[self.current_index]['url']
            webbrowser.open(url)
            
    def _save_labels(self):
        """Save all labels to disk."""
        try:
            # Save current label
            self._save_current_label()
            
            # Generate output path
            output_dir = os.path.dirname(self.dataset_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_dir}/labeled_dataset_{timestamp}.csv"
            
            # Save to CSV
            self.df.to_csv(output_path, index=False)
            logger.info(f"Labels saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save labels: {str(e)}")
            
    def run(self):
        """Start the labeling interface."""
        self.root.mainloop()

def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Label training data for AI test generator")
    parser.add_argument("dataset", help="Path to dataset CSV file")
    args = parser.parse_args()
    
    labeler = DataLabeler(args.dataset)
    labeler.run()

if __name__ == "__main__":
    main()
