import logging
from config import Config
from dataset import AudioDataset
import argparse
import os
from pipeline import DeepfakeDetectionPipeline
import torch

def main():
    """Run the complete audio deepfake detection pipeline."""
    # Configure logging
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 2. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Audio Deepfake Detection")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of data to use (e.g., 0.25 for 25%)")
    parser.add_argument("--mode", type=str,
                        choices=["train", "evaluate", "predict"], required=True,
                        help="Operation mode: train, evaluate, or predict")
    parser.add_argument("--model_prefix", type=str, default="final_model",
                        help="Prefix for saved model files")
    parser.add_argument("--audio_path", type=str,
                        help="Path to audio file for prediction (required for predict mode)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device for computation (e.g. cuda:0)")
    args = parser.parse_args()

    # 3. Disable problematic torchaudio backends
    os.environ["TORCHAUDIO_USE_SOX"] = "0"
    os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"

    # 4. Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 5. Create configuration
    config = Config()
    config.device = device
    config.data_fraction = args.data_fraction
    config.train_split = 0.8

    # 6. Choose appropriate DataLoader settings
    config.num_workers = max(1, torch.cuda.device_count() * 2)
    config.train_batch_size = getattr(config, "train_batch_size", 128)
    config.eval_batch_size = getattr(config, "eval_batch_size", 128)
    config.db_batch_size = getattr(config, "db_batch_size", 64)
    config.top_k = getattr(config, "top_k", 5)
    config.use_batch_norm = False
    config.use_layer_norm = True

    # 7. Initialize pipeline (moves all models to GPU)
    pipeline = DeepfakeDetectionPipeline(config)

    if args.mode == "train":
        # 8. Instantiate datasets once with split flag
        train_dataset = AudioDataset(config, is_train=True, split_data=True)
        val_dataset   = AudioDataset(config, is_train=False, split_data=True)

        # 9. Train with mixed-precision and GPU batching
        pipeline.train(train_dataset, val_dataset)

    elif args.mode == "evaluate":
        # 10. Load best model onto GPU
        pipeline.load_models(args.model_prefix)
        pipeline.vector_db.load()

        test_dataset = AudioDataset(config, is_train=False, split_data=False)
        metrics = pipeline.evaluate_with_metrics(test_dataset)

        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    elif args.mode == "predict":
        if not args.audio_path:
            raise ValueError("Audio path must be provided for predict mode")

        # 11. Load model & DB on GPU
        pipeline.load_models(args.model_prefix)
        pipeline.vector_db.load()

        # 12. Single-file prediction on GPU
        result = pipeline.predict(args.audio_path)
        logging.info(f"Prediction  : {result['prediction']}")
        logging.info(f"Probability : {result['probability']:.4f}")
        logging.info(f"Retrieved   : {result['retrieved_labels']}")

if __name__ == "__main__":
    main()