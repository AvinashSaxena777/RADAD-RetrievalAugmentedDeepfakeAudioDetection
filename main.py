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
    parser.add_argument("--feature_extractor", type=str, default="wav2vec2",
                        help="Feature extractor to use: whisper, wavlm, wav2vec2")
    parser.add_argument("--wandb", type=bool, default=False,
                        help="Enable or disable Weights & Biases logging")
    
    args = parser.parse_args()

    # 3. Disable problematic torchaudio backends
    os.environ["TORCHAUDIO_USE_SOX"] = "0"
    os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"

    # 4. Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logging.info("Using CPU")
    

    # 5. Create configuration
    config = Config()
    config.device = device
    config.data_fraction = args.data_fraction
    config.train_split = 0.8
    config.use_wandb = args.wandb
    config.feature_extractor_type = args.feature_extractor.lower()
    config.model_prefix = f"_{config.feature_extractor_type}" + args.model_prefix

    # 6. Choose appropriate DataLoader settings
    config.num_workers = max(1, torch.cuda.device_count() * 2)
    config.train_batch_size = getattr(config, "train_batch_size", 256)
    config.eval_batch_size = getattr(config, "eval_batch_size", 256)
    config.db_batch_size = getattr(config, "db_batch_size", 64)
    config.top_k = getattr(config, "top_k", 5)
    config.use_batch_norm = False
    config.use_layer_norm = True

    # 7. Initialize pipeline (moves all models to GPU)
    pipeline = DeepfakeDetectionPipeline(config)

    if args.mode == "train":
        train_dataset = AudioDataset(config, is_train=True, split_data=True)
        val_dataset   = AudioDataset(config, is_train=False, split_data=True)
        pipeline.print_split_stats(train_dataset, "Train")
        pipeline.print_split_stats(val_dataset,   "Val")
        pipeline.train(train_dataset, val_dataset)

    elif args.mode == "evaluate":
        config.use_wandb = False
        pipeline.load_models("final_model")
        pipeline.vector_db.load()

        test_dataset = AudioDataset(config, is_train=False, split_data=True)
        if hasattr(pipeline, "evaluate_with_metrics"):
            metrics = pipeline.evaluate_with_metrics(test_dataset)
            print("Evaluation metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value}")
        else:
            loss, acc = pipeline.evaluate(test_dataset)
            print(f"Eval Loss: {loss:.4f}, Eval Acc: {acc:.4f}")

    elif args.mode == "predict":
        if not args.audio_path:
            raise ValueError("Audio path must be provided for predict mode")
        pipeline.load_models("best_model")
        pipeline.vector_db.load()
        result = pipeline.predict(args.audio_path)
        logging.info(f"Prediction  : {result['prediction']}")
        logging.info(f"Probability(bona-fide) : {result['probability_bonafide']:.4f}")
        logging.info(f"Retrieved   : {result['retrieved_labels']}")

if __name__ == "__main__":
    main()