import logging
from config import Config
from dataset import AudioDataset

from pipeline import DeepfakeDetectionPipeline
def main():
    """Run the complete audio deepfake detection pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = Config()
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Audio Deepfake Detection")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of data to use (e.g., 0.25 for 25%)")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict"], required=True,
                        help="Operation mode: train, evaluate, or predict")
    parser.add_argument("--model_prefix", type=str, default="best_model",
                        help="Prefix for saved model files")
    parser.add_argument("--audio_path", type=str,
                        help="Path to audio file for prediction (required for predict mode)")
    
    args = parser.parse_args()
    
    # Update configuration
    config.data_fraction = args.data_fraction
    config.data_root = ""  # Update to your actual root path
    config.train_data_path = "release_in_the_wild"  # Path to your data directory
    config.test_data_path = "release_in_the_wild"  # Path to your test data
    # Create pipeline
    pipeline = DeepfakeDetectionPipeline(config)
    
    if args.mode == "train":
        # Load datasets
        train_dataset = AudioDataset(config, is_train=True)
        val_dataset = AudioDataset(config, is_train=False)
        
        # Train the model
        pipeline.train(train_dataset, val_dataset)
    
    elif args.mode == "evaluate":
        # Load model
        pipeline.load_models(args.model_prefix)
        
        # Load vector database
        pipeline.vector_db.load()
        
        # Load test dataset
        test_dataset = AudioDataset(config, is_train=False)
        
        # Evaluate the model
        loss, accuracy = pipeline.evaluate(test_dataset)
        
        logging.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    elif args.mode == "predict":
        # Check if audio path is provided
        if not args.audio_path:
            raise ValueError("Audio path must be provided for predict mode")
        
        # Load model
        pipeline.load_models(args.model_prefix)
        
        # Load vector database
        pipeline.vector_db.load()
        
        # Make prediction
        result = pipeline.predict(args.audio_path)
        
        logging.info(f"Prediction: {result['prediction']}")
        logging.info(f"Probability: {result['probability']:.4f}")
        logging.info(f"Retrieved labels: {result['retrieved_labels']}")

if __name__ == "__main__":
    main()