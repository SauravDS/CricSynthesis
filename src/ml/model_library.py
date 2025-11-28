"""
Model library management for saving and loading trained models.
"""

import os
import joblib
import json
from datetime import datetime


class ModelLibrary:
    """Manage saved fantasy cricket models."""
    
    def __init__(self, library_path='models/library'):
        self.library_path = library_path
        os.makedirs(library_path, exist_ok=True)
        self.metadata_file = os.path.join(library_path, 'models.json')
        
    def save_model(self, model, feature_names, model_info, model_name):
        """
        Save a trained model to the library.
        
        Args:
            model: Trained model object
            feature_names: List of feature names
            model_info: Dictionary with model metadata
            model_name: Unique name for this model
        """
        # Create model directory
        model_dir = os.path.join(self.library_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and features
        joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
        joblib.dump(feature_names, os.path.join(model_dir, 'features.pkl'))
        
        # Save metadata
        model_info['model_name'] = model_name
        model_info['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(os.path.join(model_dir, 'info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Update library index
        self._update_library_index(model_name, model_info)
        
        return model_dir
    
    def load_model(self, model_name):
        """
        Load a model from the library.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, feature_names, model_info)
        """
        model_dir = os.path.join(self.library_path, model_name)
        
        if not os.path.exists(model_dir):
            raise ValueError(f"Model '{model_name}' not found in library")
        
        model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        feature_names = joblib.load(os.path.join(model_dir, 'features.pkl'))
        
        with open(os.path.join(model_dir, 'info.json'), 'r') as f:
            model_info = json.load(f)
        
        return model, feature_names, model_info
    
    def list_models(self):
        """
        Get list of all saved models.
        
        Returns:
            List of dictionaries with model metadata
        """
        if not os.path.exists(self.metadata_file):
            return []
        
        with open(self.metadata_file, 'r') as f:
            library = json.load(f)
        
        return library.get('models', [])
    
    def delete_model(self, model_name):
        """Delete a model from the library."""
        import shutil
        model_dir = os.path.join(self.library_path, model_name)
        
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        # Update library index
        self._remove_from_index(model_name)
    
    def _update_library_index(self, model_name, model_info):
        """Update the library index file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                library = json.load(f)
        else:
            library = {'models': []}
        
        # Remove if exists (update)
        library['models'] = [m for m in library['models'] if m.get('model_name') != model_name]
        
        # Add new entry
        library['models'].append({
            'model_name': model_name,
            'league_name': model_info.get('league_name', 'Unknown'),
            'n_matches': model_info.get('n_matches', 0),
            'n_teams': model_info.get('n_teams', 0),
            'best_model': model_info.get('best_model', 'Unknown'),
            'r2_score': model_info['model_scores'][model_info['best_model']]['r2'],
            'saved_at': model_info.get('saved_at', '')
        })
        
        with open(self.metadata_file, 'w') as f:
            json.dump(library, f, indent=2)
    
    def _remove_from_index(self, model_name):
        """Remove model from library index."""
        if not os.path.exists(self.metadata_file):
            return
        
        with open(self.metadata_file, 'r') as f:
            library = json.load(f)
        
        library['models'] = [m for m in library['models'] if m.get('model_name') != model_name]
        
        with open(self.metadata_file, 'w') as f:
            json.dump(library, f, indent=2)
