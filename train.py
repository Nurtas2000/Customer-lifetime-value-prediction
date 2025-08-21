import argparse
import yaml
import pandas as pd
import numpy as np
from src.data.preprocessing import CLVDataPreprocessor
from src.models.maml import MAML
from src.training.meta_trainer import MetaTrainer

def main(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load and preprocess data
    df = pd.read_csv(config['data_path'])
    preprocessor = CLVDataPreprocessor()
    X, y, segments = preprocessor.fit_transform(df)
    
    # Split into meta-train and meta-test
    train_segments, test_segments = train_test_split(
        np.unique(segments), test_size=config['test_size'], random_state=config['seed'])
    
    X_train = X[np.isin(segments, train_segments)]
    y_train = y[np.isin(segments, train_segments)]
    segments_train = segments[np.isin(segments, train_segments)]
    
    # Initialize MAML
    maml = MAML(input_shape=X_train.shape[1], 
               alpha=config['alpha'], 
               beta=config['beta'])
    
    # Train
    trainer = MetaTrainer(maml, config['training'])
    loss_history = trainer.train(X_train, y_train, segments_train)
    
    # Save final model
    maml.save(config['model_save_path'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    main(args.config)
