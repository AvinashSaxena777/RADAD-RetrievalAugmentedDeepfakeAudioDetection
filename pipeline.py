import os
import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from config import Config
from dataset import AudioDataset
from segmenter import AudioSegmenter
from feature_extractor import Wav2Vec2FeatureExtractor
from vector_database import VectorDatabase
from pooling import TemporalPyramidPooling
from projection import ProjectionLayer
from detection_model import DetectionModel
# from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from feature_extractor import WhisperFeatureExtractor, WavLMFeatureExtractor, Wav2Vec2FeatureExtractor
from radad_model import RADADModel
import pandas as pd
import time
import json

# ---------------------------
# Optional: wandb helpers
# ---------------------------
def _get_wandb_api_key():
    """Try Colab secrets first, then environment variables."""
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get("WB_TOKEN")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("WB_TOKEN") or os.environ.get("WANDB_API_KEY")

def _safe_wandb_login():
    """Return (wandb_module_or_None, error_or_None)."""
    try:
        import wandb
    except Exception as e:
        return None, f"wandb import failed: {e}"
    try:
        api_key = _get_wandb_api_key()
        if api_key:
            wandb.login(key=api_key, relogin=True)
        return wandb, None
    except Exception as e:
        return None, f"wandb login failed: {e}"

def build_feature_extractor(config):
    kind = getattr(config, "feature_extractor_type", "wav2vec2").lower()
    if kind == "whisper":
        # print(f"Using {config.whisper_model_name} as Feature Extractor")
        return WhisperFeatureExtractor(config)          # you already added this earlier
    if kind == "wavlm":
        # print(f"Using {config.wavlm_model_name} as Feature Extractor")
        return WavLMFeatureExtractor(config)            # NEW
    if kind == "wav2vec2":
        # print(f"Using {config.wav2vec2_model_name} as Feature Extractor")
        return Wav2Vec2FeatureExtractor(config)         # your existing class
    raise ValueError(f"Unsupported feature_extractor_type={kind!r} (use 'wav2vec2' | 'whisper' | 'wavlm').")

# ===========================
# DeepfakeDetectionPipeline
# ===========================
class DeepfakeDetectionPipeline:
    """Main pipeline for audio deepfake detection with single-GPU optimizations."""

    def __init__(self, config: 'Config'):
        self.config = config
        self.device = config.device

        # 1. Audio segmenter (CPU)
        self.audio_segmenter = AudioSegmenter(config)

        # 2. Feature extractor (kept on CPU internally, model lives on GPU)
        self.feature_extractor = build_feature_extractor(config)

        # 3. Set feature_dim in config
        if not hasattr(config, "feature_dim"):
            config.feature_dim = self.feature_extractor.feature_dim
            print(f"Feature dimension set to: {config.feature_dim}")

        # 4. TPP and vector DB
        self.tpp = TemporalPyramidPooling(config)
        self.vector_db = VectorDatabase(config)

        # 5. Single encapsulated model (Projection + Fuse + Detection)
        self.radad_model = RADADModel(config, self.tpp.get_output_dim()).to(self.device)

        # 6. Optimizers & AMP scaler (keep functionality: 3 separate optimizers)
        self.projection_optimizer = torch.optim.Adam(
            self.radad_model.projection_layer.parameters(),
            lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.fuse_optimizer = torch.optim.Adam(
            self.radad_model.fuse.parameters(),
            lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.detection_optimizer  = torch.optim.Adam(
            self.radad_model.detection_model.parameters(),
            lr=config.learning_rate, weight_decay=config.weight_decay
        )

        self.scaler = GradScaler("cuda")

        # Track training file IDs for leakage checks
        self.training_file_ids = set()

        # Wandb plumbing
        self._wandb_enabled = bool(getattr(config, "use_wandb", False))
        self._wandb = None
        self._wandb_run = None
        self._global_step = 0  # per-batch step counter

        # ---- metrics tracking (for CSV + plots) ----
        self._metrics_rows: List[dict] = []   # one dict per epoch
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        self._train_accs: List[float] = []
        self._val_accs: List[float] = []

        # best trackers for summary.json
        self._best_by_val_loss = {"epoch": None, "val_loss": np.inf}
        self._best_by_eer = {"epoch": None, "eer_percent": np.inf}

    # ----------------- Metric utilities -----------------
    @staticmethod
    def compute_pos_weight_from_dataset(dataset) -> float:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
        pos = 0
        neg = 0
        for b in loader:
            y = b['label']
            if isinstance(y, torch.Tensor):
                y = y.float()
            else:
                y = torch.tensor(y, dtype=torch.float32)
            pos += (y > 0.5).sum().item()   # 1 = bona-fide
            neg += (y <= 0.5).sum().item()  # 0 = spoof
        pos_weight = (neg + 1.0) / (pos + 1.0)  # smoothed
        if not np.isfinite(pos_weight):
            pos_weight = 1.0
        return float(np.clip(pos_weight, 0.1, 10.0))

    @staticmethod
    def _compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """
        EER (%) and threshold where FPR == FNR.
        labels: 1 = bona-fide (positive), 0 = spoof (negative)
        scores: higher => more bona-fide (we use logits)
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        P = len(pos)
        N = len(neg)
        if P == 0 or N == 0:
            return float("nan"), float("nan")

        thrs = np.r_[-np.inf, np.unique(scores), np.inf]
        pos_sorted = np.sort(pos)
        neg_sorted = np.sort(neg)
        fnr = np.searchsorted(pos_sorted, thrs, side='left') / max(P, 1)  # miss bona-fide
        fpr = (N - np.searchsorted(neg_sorted, thrs, side='left')) / max(N, 1)  # accept spoof
        diff = np.abs(fnr - fpr)
        k = int(np.argmin(diff))
        eer = (fnr[k] + fpr[k]) / 2.0
        return float(eer * 100.0), float(thrs[k])

    @staticmethod
    def _compute_macro_eer_by_group(scores: np.ndarray, labels: np.ndarray, groups: List[str]) -> float:
        """Average EER across groups (e.g., speakers)."""
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        eers = []
        for g in uniq:
            m = groups == g
            y = labels[m]
            s = scores[m]
            if (y == 1).any() and (y == 0).any():
                eer_g, _ = DeepfakeDetectionPipeline._compute_eer(s, y)
                if np.isfinite(eer_g):
                    eers.append(eer_g)
        if len(eers) == 0:
            return float("nan")
        return float(np.mean(eers))

    @staticmethod
    def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute FPR, TPR, thresholds (like sklearn.roc_curve but tiny & dependency-free).
        Positive class is bona-fide (label 1), higher score => more bona-fide.
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        # sort by decreasing score
        order = np.argsort(-scores)
        scores = scores[order]
        labels = labels[order]

        P = (labels == 1).sum()
        N = (labels == 0).sum()
        if P == 0 or N == 0:
            return np.array([0,1]), np.array([0,1]), np.array([np.inf, -np.inf])

        tps = np.cumsum(labels == 1)
        fps = np.cumsum(labels == 0)

        # thresholds at each unique score (keep first occurrence)
        distinct = np.r_[True, scores[1:] != scores[:-1]]
        tps = tps[distinct]
        fps = fps[distinct]
        thresholds = scores[distinct]

        tpr = tps / P
        fpr = fps / N

        # prepend (0,0) and append (1,1)
        tpr = np.r_[0.0, tpr, 1.0]
        fpr = np.r_[0.0, fpr, 1.0]
        thresholds = np.r_[thresholds[0] + 1e-6, thresholds, thresholds[-1] - 1e-6]
        return fpr, tpr, thresholds

    @staticmethod
    def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
        """Trapezoidal area under ROC."""
        return float(np.trapz(tpr, fpr))

    @staticmethod
    def _probit(x: np.ndarray) -> np.ndarray:
        """Inverse of normal CDF (for DET axes). Fallback if scipy is absent."""
        try:
            from scipy.stats import norm  # type: ignore
            return norm.ppf(x)
        except Exception:
            # rational approximation (Acklam's) – good enough for plots
            eps = 1e-9
            x = np.clip(x, eps, 1 - eps)
            a1 = -39.69683028665376; a2 = 220.9460984245205; a3 = -275.9285104469687
            a4 = 138.3577518672690; a5 = -30.66479806614716; a6 = 2.506628277459239
            b1 = -54.47609879822406; b2 = 161.5858368580409; b3 = -155.6989798598866
            b4 = 66.80131188771972; b5 = -13.28068155288572
            c1 = -0.007784894002430293; c2 = -0.3223964580411365
            c3 = -2.400758277161838; c4 = -2.549732539343734
            c5 = 4.374664141464968; c6 = 2.938163982698783
            d1 = 0.007784695709041462; d2 = 0.3224671290700398
            d3 = 2.445134137142996; d4 = 3.754408661907416
            plow = 0.02425; phigh = 1 - plow
            q = np.empty_like(x)
            m1 = x < plow
            m2 = (x >= plow) & (x <= phigh)
            m3 = x > phigh
            if np.any(m1):
                q1 = np.sqrt(-2*np.log(x[m1]))
                q[m1] = (((((c1*q1 + c2)*q1 + c3)*q1 + c4)*q1 + c5)*q1 + c6) / \
                        ((((d1*q1 + d2)*q1 + d3)*q1 + d4)*q1 + 1)
                q[m1] *= -1
            if np.any(m2):
                q2 = x[m2] - 0.5
                r = q2*q2
                q[m2] = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q2 / \
                         (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1)
            if np.any(m3):
                q3 = np.sqrt(-2*np.log(1 - x[m3]))
                q[m3] = (((((c1*q3 + c2)*q3 + c3)*q3 + c4)*q3 + c5)*q3 + c6) / \
                        ((((d1*q3 + d2)*q3 + d3)*q3 + d4)*q3 + 1)
            return q

    @staticmethod
    def _compute_min_tDCF(
        cm_scores: np.ndarray,
        labels: np.ndarray,
        asv_params: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Normalized min t-DCF for CM before ASV.
        Requires asv_params (see earlier messages).
        """
        req = {"P_miss_asv","P_fa_asv","P_fa_spoof_asv",
               "C_miss_asv","C_fa_asv","C_miss_cm","C_fa_cm",
               "pi_tar","pi_non","pi_spoof"}
        if asv_params is None or any(k not in asv_params for k in req):
            return float("nan"), float("nan")

        P_miss_asv = float(asv_params["P_miss_asv"])
        P_fa_asv   = float(asv_params["P_fa_asv"])
        P_fa_spoof_asv = float(asv_params["P_fa_spoof_asv"])
        C_miss_asv = float(asv_params["C_miss_asv"])
        C_fa_asv   = float(asv_params["C_fa_asv"])
        C_miss_cm  = float(asv_params["C_miss_cm"])
        C_fa_cm    = float(asv_params["C_fa_cm"])
        pi_tar     = float(asv_params["pi_tar"])
        pi_non     = float(asv_params["pi_non"])
        pi_spoof   = float(asv_params["pi_spoof"])

        C_def = min(C_miss_asv * pi_tar, C_fa_asv * pi_non)
        if C_def <= 0:
            return float("nan"), float("nan")

        thrs = np.r_[-np.inf, np.unique(cm_scores), np.inf]
        bf = cm_scores[labels == 1]
        sp = cm_scores[labels == 0]
        B = len(bf); S = len(sp)
        if B == 0 or S == 0:
            return float("nan"), float("nan")
        bf_sorted = np.sort(bf)
        sp_sorted = np.sort(sp)
        Pmiss_cm = np.searchsorted(bf_sorted, thrs, side="left") / max(B, 1)
        Pfa_cm   = (S - np.searchsorted(sp_sorted, thrs, side="left")) / max(S, 1)

        tdcf = (
            C_miss_asv * pi_tar * P_miss_asv
          + C_fa_asv   * pi_non * P_fa_asv
          + C_fa_cm    * pi_spoof * (1.0 - Pmiss_cm) * P_fa_spoof_asv
          + C_miss_cm  * pi_tar   * Pmiss_cm
        ) / C_def

        k = int(np.argmin(tdcf))
        return float(tdcf[k]), float(thrs[k])

    # ------------- wandb helpers -------------
    def _wandb_init(self, train_dataset=None, val_dataset=None, pos_weight=None):
        if not self._wandb_enabled:
            return
        self._wandb, err = _safe_wandb_login()
        if self._wandb is None:
            logging.warning(f"wandb disabled (login/import problem): {err}")
            self._wandb_enabled = False
            return
        cfg = {
            "device": str(self.device),
            "feature_dim": getattr(self.config, "feature_dim", None),
            "tpp_output_dim": self.tpp.get_output_dim(),
            "projection_output_dim": getattr(self.config, "projection_output_dim", None),
            "detection_hidden_dims": getattr(self.config, "detection_hidden_dims", []),
            "learning_rate": getattr(self.config, "learning_rate", None),
            "weight_decay": getattr(self.config, "weight_decay", None),
            "train_batch_size": getattr(self.config, "train_batch_size", None),
            "eval_batch_size": getattr(self.config, "eval_batch_size", None),
            "db_batch_size": getattr(self.config, "db_batch_size", None),
            "num_epochs": getattr(self.config, "num_epochs", None),
            "top_k": getattr(self.config, "top_k", None),
            "segment_length": getattr(self.config, "segment_length", None),
            "sample_rate": getattr(self.config, "sample_rate", None),
            "use_batch_norm": getattr(self.config, "use_batch_norm", None),
            "use_layer_norm": getattr(self.config, "use_layer_norm", None),
            "use_mixed_precision": getattr(self.config, "use_mixed_precision", None),
            "pos_weight": pos_weight,
            "data_fraction": getattr(self.config, "data_fraction", None),
            "train_split": getattr(self.config, "train_split", None),
            "vector_db_index_type": getattr(self.config, "vector_db_index_type", None),
        }
        run_name = f"{os.environ.get('USER','colab')}-{os.environ.get('HOSTNAME','node')}-{os.getpid()}"
        project = os.environ.get("WANDB_PROJECT", "deepfake-audio-raf")
        self._wandb_run = self._wandb.init(project=project, name=run_name, config=cfg)
        self._wandb.watch(self.radad_model, log="all", log_freq=200)
        logging.info("wandb run initialized")

    def _wandb_log(self, metrics: dict):
        if self._wandb_enabled and self._wandb_run is not None:
            try:
                self._wandb.log(metrics, step=self._global_step)
            except Exception as e:
                logging.warning(f"wandb.log failed: {e}")

    def _wandb_finish(self):
        if self._wandb_enabled and self._wandb_run is not None:
            try:
                models_dir = os.path.join(self.config.data_root, "models")
                if os.path.isdir(models_dir):
                    artifact = self._wandb.Artifact("deepfake_models", type="model")
                    for fname in os.listdir(models_dir):
                        if fname.endswith(".pt"):
                            artifact.add_file(os.path.join(models_dir, fname))
                    self._wandb_run.log_artifact(artifact)
            except Exception as e:
                logging.warning(f"W&B artifact logging failed: {e}")
            try:
                self._wandb.finish()
            except Exception:
                pass
            self._wandb_run = None

    # ------------- core pipeline -------------
    def process_audio_batch(self,
                            audio_paths: List[str],
                            audio_dataset: 'AudioDataset') -> torch.Tensor:
        segments_batch = []
        for path in audio_paths:
            wav = audio_dataset.load_audio(path)
            if wav is None:
                raise RuntimeError(f"Failed to load '{path}'")
            segments_batch.append(self.audio_segmenter.segment_audio(wav))

        feature_batches = []
        for segments in segments_batch:
            cpu_tensors = self.feature_extractor.extract_features(segments)
            gpu_tensors = [t.to(self.device) for t in cpu_tensors]
            feature_batches.append(gpu_tensors)

        pooled = []
        for gpu_list in feature_batches:
            seg_pooled = [self.tpp.pool_features(ft) for ft in gpu_list]
            mean_pooled = torch.mean(torch.stack(seg_pooled), dim=0)
            pooled.append(mean_pooled)

        return torch.stack(pooled)  # [batch, tpp_output_dim]

    def build_vector_database(self, train_dataset: 'AudioDataset'):
        logging.info("Building vector database from TRAINING data...")
        loader = DataLoader(train_dataset,
                            batch_size=self.config.db_batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=self.config.num_workers)

        vectors, paths, labels, metadata = [], [], [], {'speaker_id': []}
        self.training_file_ids.clear()
        audio_ds = AudioDataset(self.config, is_train=False)

        for batch in tqdm(loader, desc="Vector DB Build"):
            audio_paths = batch['path']
            batch_vecs = self.process_audio_batch(audio_paths, audio_ds).cpu().numpy()

            metas = batch['metadata']
            batch_size = len(audio_paths)
            batch_speakers = self._extract_batch_speakers(metas, batch_size)

            for vec, path, lbl, spk in zip(batch_vecs, audio_paths, batch['label'], batch_speakers):
                file_id = os.path.basename(path)
                self.training_file_ids.add(file_id)
                vectors.append(vec)
                paths.append(path)
                labels.append(lbl)
                metadata['speaker_id'].append(spk)

        vectors = np.vstack(vectors)
        self.vector_db.add_vectors(vectors, paths, labels, metadata)
        self.vector_db.save()
        logging.info(f"Built vector DB ({len(vectors)} samples)")

    def retrieve_similar_vectors(self, query_vectors: torch.Tensor,
                                 query_paths: Optional[List[str]] = None,
                                 exclude_self: bool = True,
                                 return_info: bool = False,
                                 return_distances: bool = False):
        import numpy as np, os

        q_np = query_vectors.detach().cpu().numpy().astype(np.float32)  # [B, D]
        B = q_np.shape[0]
        K = int(self.config.top_k)
        D = self.tpp.get_output_dim()

        exclude_ids = set()
        if exclude_self and query_paths is not None:
            exclude_ids = {os.path.basename(p) for p in query_paths}

        if (self.vector_db.index is None) or (getattr(self.vector_db.index, "ntotal", 0) == 0):
            vec_tensor = torch.zeros(B, K, D, device=self.device)
            lbl_tensor = torch.zeros(B, K, device=self.device)
            empty_paths = [[""] * K for _ in range(B)]
            dist_tensor = torch.full((B, K), float("nan"), device=self.device)
            if return_info and return_distances:
                return vec_tensor, lbl_tensor, empty_paths, dist_tensor
            if return_info:
                return vec_tensor, lbl_tensor, empty_paths
            if return_distances:
                return vec_tensor, lbl_tensor, dist_tensor
            return vec_tensor, lbl_tensor

        k_search = K + (10 if exclude_self else 0)
        try:
            dists, idxs = self.vector_db.search_batch(q_np, k=k_search)
        except Exception:
            dists = np.zeros((B, 0), dtype=np.float32)
            idxs  = np.zeros((B, 0), dtype=np.int64)

        if idxs is None or idxs.ndim != 2 or idxs.shape[0] != B:
            idxs = np.zeros((B, 0), dtype=np.int64)
            dists = np.zeros((B, 0), dtype=np.float32)

        all_vecs, all_lbls, all_paths, all_dists = [], [], [], []

        for row_idx, (row_inds, row_dists) in enumerate(zip(idxs, dists)):
            chosen_vecs, chosen_lbls, chosen_paths, chosen_dists = [], [], [], []
            for ii, dd in zip(row_inds, row_dists):
                ii = int(ii)
                fname = os.path.basename(self.vector_db.vector_paths[ii])
                if exclude_self:
                    if query_paths is not None:
                        if fname in exclude_ids:
                            continue
                    else:
                        if fname in getattr(self, "training_file_ids", set()):
                            continue
                vec = self.vector_db.index.reconstruct(ii)
                chosen_vecs.append(vec)
                chosen_lbls.append(self.vector_db.vector_labels[ii])
                chosen_paths.append(self.vector_db.vector_paths[ii])
                chosen_dists.append(float(dd))
                if len(chosen_vecs) == K:
                    break

            while len(chosen_vecs) < K:
                chosen_vecs.append(np.zeros(D, dtype=np.float32))
                chosen_lbls.append(0.0)
                chosen_paths.append("")
                chosen_dists.append(float("nan"))

            all_vecs.append(chosen_vecs)
            all_lbls.append(chosen_lbls)
            all_paths.append(chosen_paths)
            all_dists.append(chosen_dists)

        vec_tensor = torch.as_tensor(np.stack(all_vecs, axis=0), device=self.device, dtype=torch.float32)
        lbl_tensor = torch.as_tensor(np.stack(all_lbls, axis=0), device=self.device, dtype=torch.float32)
        dist_tensor = torch.as_tensor(np.stack(all_dists, axis=0), device=self.device, dtype=torch.float32)

        if return_info and return_distances:
            return vec_tensor, lbl_tensor, all_paths, dist_tensor
        if return_info:
            return vec_tensor, lbl_tensor, all_paths
        if return_distances:
            return vec_tensor, lbl_tensor, dist_tensor
        return vec_tensor, lbl_tensor

    # ---------- metrics I/O & plots ----------
    def _ensure_criterion(self, dataset: 'AudioDataset'):
      """Make sure self.criterion exists; initialize with pos_weight from dataset if needed."""
      if getattr(self, "criterion", None) is None:
          pos_weight = self.compute_pos_weight_from_dataset(dataset)
          self.criterion = torch.nn.BCEWithLogitsLoss(
              pos_weight=torch.tensor([pos_weight], device=self.device, dtype=torch.float32)
          )
          logging.info(f"[evaluate] Initialized criterion with pos_weight={pos_weight:.3f}")


    def _metrics_csv_path(self) -> str:
        return os.path.join(self.config.data_root, "metrics.csv")

    def save_metrics_csv(self):
        if not self._metrics_rows:
            return
        df = pd.DataFrame(self._metrics_rows)
        csv_path = self._metrics_csv_path()
        os.makedirs(self.config.data_root, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved metrics CSV to {csv_path}")

    def _extract_batch_speakers(self, metas, batch_size: int) -> List[str]:
        """
        Return a list of length batch_size with speaker ids.
        Handles both list-of-dicts and dict-of-lists (default_collate) cases.
        """
        speakers: List[str] = []

        if isinstance(metas, list):
            for m in metas:
                if isinstance(m, dict):
                    speakers.append(str(m.get('speaker_id', 'unknown')))
                else:
                    speakers.append(str(m))
        elif isinstance(metas, dict):
            vals = metas.get('speaker_id', None)
            if isinstance(vals, (list, tuple)):
                speakers = [str(v) for v in vals]
            elif vals is not None:
                speakers = [str(vals)] * batch_size

        if len(speakers) < batch_size:
            speakers = speakers + (["unknown"] * (batch_size - len(speakers)))
        elif len(speakers) > batch_size:
            speakers = speakers[:batch_size]
        return speakers

    def plot_training_curves(self, save_dir: Optional[str] = None, log_to_wandb: bool = True):
        if save_dir is None:
            save_dir = os.path.join(self.config.data_root, "plots")
        os.makedirs(save_dir, exist_ok=True)

        epochs = list(range(1, len(self._train_losses) + 1))

        # Loss curve
        plt.figure()
        plt.plot(epochs, self._train_losses, label="Train Loss")
        if len(self._val_losses) == len(epochs):
            plt.plot(epochs, self._val_losses, label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train / Val Loss"); plt.legend()
        loss_path = os.path.join(save_dir, "train_val_loss.png")
        plt.savefig(loss_path, bbox_inches="tight"); plt.close()

        # Accuracy curve
        plt.figure()
        plt.plot(epochs, self._train_accs, label="Train Acc")
        if len(self._val_accs) == len(epochs):
            plt.plot(epochs, self._val_accs, label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Train / Val Accuracy"); plt.legend()
        acc_path = os.path.join(save_dir, "train_val_acc.png")
        plt.savefig(acc_path, bbox_inches="tight"); plt.close()

        logging.info(f"Saved plots: {loss_path}, {acc_path}")

        if self._wandb_enabled and self._wandb_run is not None and log_to_wandb:
            try:
                self._wandb.log({
                    "plots/train_val_loss": self._wandb.Image(loss_path),
                    "plots/train_val_acc": self._wandb.Image(acc_path)
                })
            except Exception as e:
                logging.warning(f"wandb image log failed: {e}")

    def _plot_and_save_roc_det(self, scores: np.ndarray, labels: np.ndarray, epoch: int):
        """
        Save ROC + DET plots and CSV points; log to wandb if enabled.
        """
        plots_dir = os.path.join(self.config.data_root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        tables_dir = os.path.join(self.config.data_root, "tables")
        os.makedirs(tables_dir, exist_ok=True)

        # ROC
        fpr, tpr, thr = self._roc_curve(scores, labels)
        auc = self._auc(fpr, tpr)

        # save CSV
        roc_df = pd.DataFrame({"threshold": thr, "fpr": fpr, "tpr": tpr})
        roc_csv = os.path.join(tables_dir, f"roc_points_epoch{epoch}.csv")
        roc_df.to_csv(roc_csv, index=False)

        # plot ROC
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR (spoof accepted)")
        plt.ylabel("TPR (bona-fide accepted)")
        plt.title(f"ROC (Epoch {epoch}) AUC={auc:.4f}")
        roc_png = os.path.join(plots_dir, f"roc_epoch{epoch}.png")
        plt.savefig(roc_png, bbox_inches="tight"); plt.close()

        # DET (probit axes)
        fnr = 1.0 - tpr
        try:
            x = self._probit(fpr)
            y = self._probit(fnr)
            plt.figure()
            plt.plot(x, y)
            plt.xlabel("norminv(FPR)")
            plt.ylabel("norminv(FNR)")
            plt.title(f"DET (Epoch {epoch})")
            det_png = os.path.join(plots_dir, f"det_epoch{epoch}.png")
            plt.savefig(det_png, bbox_inches="tight"); plt.close()
        except Exception:
            plt.figure()
            plt.plot(fpr, fnr)
            plt.xlabel("FPR")
            plt.ylabel("FNR")
            plt.title(f"DET (linear fallback) Epoch {epoch}")
            det_png = os.path.join(plots_dir, f"det_epoch{epoch}.png")
            plt.savefig(det_png, bbox_inches="tight"); plt.close()

        det_df = pd.DataFrame({"threshold": thr, "fpr": fpr, "fnr": fnr})
        det_csv = os.path.join(tables_dir, f"det_points_epoch{epoch}.csv")
        det_df.to_csv(det_csv, index=False)

        # wandb
        self._wandb_log({
            "curves/auc": float(auc),
        })
        if self._wandb_enabled and self._wandb_run is not None:
            try:
                self._wandb.log({
                    "plots/roc": self._wandb.Image(roc_png),
                    "plots/det": self._wandb.Image(det_png),
                })
                art = self._wandb.Artifact(f"epoch{epoch}_curves", type="curves")
                art.add_file(roc_csv)
                art.add_file(det_csv)
                self._wandb_run.log_artifact(art)
            except Exception as e:
                logging.warning(f"W&B curve logging failed: {e}")

        return float(auc)

    # ------------- eval helpers that also collect scores -------------
    def evaluate_with_scores(self, val_dataset: 'AudioDataset') -> Tuple[float, float, np.ndarray, np.ndarray, List[str]]:
        """
        Evaluate and also return (scores, labels, speakers) for extra metrics.
        scores are raw logits where larger ⇒ bona-fide.
        """
        loader = DataLoader(val_dataset,
                            batch_size=self.config.eval_batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=self.config.num_workers)

        self.radad_model.eval()
        self._ensure_criterion(val_dataset)

        total_loss, correct, total = 0., 0, 0
        all_scores: List[float] = []
        all_labels: List[int] = []
        all_speakers: List[str] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                paths, lbls = batch['path'], batch['label']
                tpp = self.process_audio_batch(paths, val_dataset)
                vecs, _ = self.retrieve_similar_vectors(tpp, query_paths=paths, exclude_self=True)

                with autocast(self.device.type):
                    if vecs.ndim == 1:
                        vecs = vecs.unsqueeze(0).unsqueeze(1)
                    elif vecs.ndim == 2:
                        vecs = vecs.unsqueeze(1)
                    logits = self.radad_model(vecs, tpp)
                    if logits.ndim == 1:
                        logits = logits.unsqueeze(-1)
                    labels = lbls.to(self.device).to(dtype=logits.dtype).view_as(logits)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item() * lbls.size(0)
                preds = (logits > 0).to(labels.dtype)
                correct += (preds == labels).sum().item()
                total   += labels.numel()

                all_scores.extend(logits.detach().cpu().view(-1).tolist())
                all_labels.extend(lbls.detach().cpu().view(-1).tolist())

                metas = batch.get('metadata', None)
                batch_size = lbls.shape[0]
                if metas is None:
                    batch_speakers = ["unknown"] * batch_size
                else:
                    batch_speakers = self._extract_batch_speakers(metas, batch_size)
                all_speakers.extend(batch_speakers)

                # Guard if collate produced a mismatch for speakers/labels
                if len(all_speakers) != len(all_labels):
                    logging.warning(
                        f"Speaker/label length mismatch after batch: "
                        f"{len(all_speakers)} vs {len(all_labels)}. Padding with 'unknown'."
                    )
                    while len(all_speakers) < len(all_labels):
                        all_speakers.append("unknown")
                    if len(all_speakers) > len(all_labels):
                        all_speakers = all_speakers[:len(all_labels)]

        val_loss = total_loss / max(total, 1)
        val_acc  = correct / max(total, 1)
        return val_loss, val_acc, np.asarray(all_scores, dtype=np.float64), np.asarray(all_labels, dtype=np.int32), all_speakers


    # ------------- train / eval / predict -------------
    def train(self, train_dataset: 'AudioDataset', val_dataset: Optional['AudioDataset'] = None):
        """Single-GPU, mixed-precision train loop."""
        if val_dataset:
            self.validate_no_leakage(val_dataset)
        if self.vector_db.index is None:
            self.build_vector_database(train_dataset)

        pos_weight = self.compute_pos_weight_from_dataset(train_dataset)
        self.criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self.device, dtype=torch.float32)
        )
        logging.info(f"Using pos_weight={pos_weight:.3f} for BCEWithLogitsLoss")

        # wandb init
        self._wandb_init(train_dataset, val_dataset, pos_weight=pos_weight)

        loader = DataLoader(train_dataset,
                            batch_size=self.config.train_batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=self.config.num_workers)

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            self.radad_model.train()
            epoch_loss, correct, total = 0., 0, 0

            # running means for diagnostics
            sum_nnz_rate = 0.0
            sum_gn_proj = 0.0
            sum_gn_fuse = 0.0
            sum_gn_det  = 0.0
            num_batches = 0

            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                paths, lbls = batch['path'], batch['label']
                tpp = self.process_audio_batch(paths, train_dataset)  # [B, dim]
                vecs, _ = self.retrieve_similar_vectors(tpp, query_paths=paths, exclude_self=True)

                if torch.isnan(tpp).any():
                    raise RuntimeError("NaNs detected in TPP embeddings")
                if torch.isnan(vecs).any():
                    logging.warning("NaNs in retrieved neighbor vectors; replacing with zeros.")
                    vecs = torch.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)

                with torch.no_grad():
                    nnz = (vecs.abs().sum(dim=-1) > 0).float().mean().item()

                with autocast(self.device.type):
                    if vecs.ndim == 1: vecs = vecs.unsqueeze(0).unsqueeze(1)
                    elif vecs.ndim == 2: vecs = vecs.unsqueeze(1)
                    logits = self.radad_model(vecs, tpp)
                    if logits.ndim == 1:
                        logits = logits.unsqueeze(-1)
                    labels = lbls.to(self.device).to(dtype=logits.dtype).view_as(logits)
                    loss = self.criterion(logits, labels)

                self.projection_optimizer.zero_grad()
                self.fuse_optimizer.zero_grad()
                self.detection_optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.projection_optimizer)
                self.scaler.unscale_(self.fuse_optimizer)
                self.scaler.unscale_(self.detection_optimizer)
                gnorm_proj = float(torch.nn.utils.clip_grad_norm_(self.radad_model.projection_layer.parameters(), max_norm=1.0))
                gnorm_fuse = float(torch.nn.utils.clip_grad_norm_(self.radad_model.fuse.parameters(), max_norm=1.0))
                gnorm_det  = float(torch.nn.utils.clip_grad_norm_(self.radad_model.detection_model.parameters(), max_norm=1.0))

                self.scaler.step(self.projection_optimizer)
                self.scaler.step(self.fuse_optimizer)
                self.scaler.step(self.detection_optimizer)
                self.scaler.update()

                epoch_loss += loss.item() * lbls.size(0)
                preds = (logits > 0).to(labels.dtype)
                correct += (preds == labels).sum().item()
                total   += labels.numel()

                sum_nnz_rate += nnz
                sum_gn_proj += gnorm_proj
                sum_gn_fuse += gnorm_fuse
                sum_gn_det  += gnorm_det
                num_batches += 1

                self._wandb_log({
                    "train/batch_loss": float(loss.item()),
                    "train/nnz_neighbor_rate": float(nnz),
                    "grad_norm/projection": gnorm_proj,
                    "grad_norm/fuse": gnorm_fuse,
                    "grad_norm/detection": gnorm_det,
                    "lr/projection": float(self.projection_optimizer.param_groups[0]["lr"]),
                    "lr/fuse": float(self.fuse_optimizer.param_groups[0]["lr"]),
                    "lr/detection": float(self.detection_optimizer.param_groups[0]["lr"]),
                })
                self._global_step += 1

            train_loss = epoch_loss / max(total, 1)
            train_acc  = correct / max(total, 1)
            self._train_losses.append(train_loss)
            self._train_accs.append(train_acc)

            # ----- Validation + extra metrics -----
            if val_dataset:
                val_loss, val_acc, v_scores, v_labels, v_speakers = self.evaluate_with_scores(val_dataset)
                self._val_losses.append(val_loss)
                self._val_accs.append(val_acc)

                eer, eer_thr = self._compute_eer(v_scores, v_labels)
                macro_eer = self._compute_macro_eer_by_group(v_scores, v_labels, v_speakers)
                asv_params = getattr(self.config, "asv_params", None)
                min_tdcf, tdcf_thr = self._compute_min_tDCF(v_scores, v_labels, asv_params)
                auc = self._plot_and_save_roc_det(v_scores, v_labels, epoch=epoch+1)

                if np.isfinite(val_loss) and val_loss < self._best_by_val_loss["val_loss"]:
                    self._best_by_val_loss = {"epoch": epoch+1, "val_loss": float(val_loss)}
                if np.isfinite(eer) and eer < self._best_by_eer["eer_percent"]:
                    self._best_by_eer = {"epoch": epoch+1, "eer_percent": float(eer)}

                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc:{val_acc:.4f} | "
                      f"AUC: {auc:.4f}, EER: {eer:.2f}% (thr={eer_thr:.4f}), "
                      f"Macro EER: {macro_eer:.2f}%, min t-DCF: {min_tdcf if np.isfinite(min_tdcf) else float('nan'):.4f}")

                self._wandb_log({
                    "epoch": epoch + 1,
                    "train/loss": float(train_loss),
                    "train/acc": float(train_acc),
                    "val/loss": float(val_loss),
                    "val/acc": float(val_acc),
                    "metrics/auc": float(auc),
                    "metrics/eer_percent": float(eer),
                    "metrics/macro_eer_percent": float(macro_eer),
                    "metrics/eer_threshold": float(eer_thr),
                    "metrics/min_tDCF": float(min_tdcf) if np.isfinite(min_tdcf) else None,
                    "metrics/min_tDCF_threshold": float(tdcf_thr) if np.isfinite(min_tdcf) else None,
                })
            else:
                print(f"Epoch {epoch+1}: Train {train_loss:.4f}/{train_acc:.4f}")
                auc = eer = macro_eer = min_tdcf = eer_thr = tdcf_thr = float("nan")
                self._wandb_log({
                    "epoch": epoch + 1,
                    "train/loss": float(train_loss),
                    "train/acc": float(train_acc),
                })

            # epoch diagnostics → metrics.csv row
            avg_nnz = (sum_nnz_rate / num_batches) if num_batches else float("nan")
            avg_gn_proj = (sum_gn_proj / num_batches) if num_batches else float("nan")
            avg_gn_fuse = (sum_gn_fuse / num_batches) if num_batches else float("nan")
            avg_gn_det  = (sum_gn_det  / num_batches) if num_batches else float("nan")
            lr_p = float(self.projection_optimizer.param_groups[0]["lr"])
            lr_f = float(self.fuse_optimizer.param_groups[0]["lr"])
            lr_d = float(self.detection_optimizer.param_groups[0]["lr"])
            epoch_time = time.time() - epoch_start

            row = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(self._val_losses[-1]) if val_dataset else None,
                "val_acc": float(self._val_accs[-1]) if val_dataset else None,
                "auc": float(auc) if np.isfinite(auc) else None,
                "eer_percent": float(eer) if np.isfinite(eer) else None,
                "pooled_eer_percent": float(eer) if np.isfinite(eer) else None,
                "macro_eer_percent": float(macro_eer) if np.isfinite(macro_eer) else None,
                "eer_threshold": float(eer_thr) if np.isfinite(eer) else None,
                "min_tDCF": float(min_tdcf) if np.isfinite(min_tdcf) else None,
                "min_tDCF_threshold": float(tdcf_thr) if np.isfinite(min_tdcf) else None,
                "avg_nnz_neighbor_rate": float(avg_nnz),
                "avg_grad_norm_projection": float(avg_gn_proj),
                "avg_grad_norm_fuse": float(avg_gn_fuse),
                "avg_grad_norm_detection": float(avg_gn_det),
                "lr_projection": lr_p,
                "lr_fuse": lr_f,
                "lr_detection": lr_d,
                "pos_weight": float(pos_weight),
                "epoch_time_sec": float(epoch_time),
                "top_k": int(getattr(self.config, "top_k", 0)),
                "batch_size": int(getattr(self.config, "train_batch_size", 0)),
            }
            self._metrics_rows.append(row)
            self.save_metrics_csv()
            self.plot_training_curves()

        self.save_models("final_model")
        self._save_summary_json()
        self._wandb_finish()

    def _save_summary_json(self):
        """Write a compact summary.json with best epochs and final metrics."""
        os.makedirs(self.config.data_root, exist_ok=True)
        summary = {
            "final_epoch": len(self._metrics_rows),
            "best_by_val_loss": self._best_by_val_loss,
            "best_by_eer": self._best_by_eer,
            "last_row": self._metrics_rows[-1] if self._metrics_rows else {},
        }
        path = os.path.join(self.config.data_root, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Saved summary to {path}")
        self._wandb_log({"summary/json": summary})

    def evaluate(self, val_dataset: 'AudioDataset', save_curves: bool = True) -> Tuple[float, float]:
        """
        Standalone evaluation that mirrors train-time metrics:
        - ensures criterion is set
        - computes loss/acc
        - computes AUC, EER, Macro-EER, min t-DCF (if ASV params set)
        - logs to W&B (if enabled)
        - appends a row to metrics.csv
        - optionally saves ROC/DET plots
        Returns (val_loss, val_acc) for backward compatibility.
        """
        # Make sure loss exists
        self._ensure_criterion(val_dataset)
        self.radad_model.eval()

        # Core metrics + scores
        val_loss, val_acc, scores, labels, speakers = self.evaluate_with_scores(val_dataset)

        # Extra metrics
        eer, eer_thr = self._compute_eer(scores, labels)
        macro_eer    = self._compute_macro_eer_by_group(scores, labels, speakers)
        asv_params   = getattr(self.config, "asv_params", None)
        min_tdcf, tdcf_thr = self._compute_min_tDCF(scores, labels, asv_params)
        auc = self._plot_and_save_roc_det(scores, labels, epoch=0) if save_curves else float("nan")

        print(
            f"Eval Loss: {val_loss:.4f}, Eval Acc: {val_acc:.4f} | "
            f"AUC: {auc:.4f}, EER: {eer:.2f}% (thr={eer_thr:.4f}), "
            f"Macro EER: {macro_eer:.2f}%, min t-DCF: {min_tdcf if np.isfinite(min_tdcf) else float('nan'):.4f}"
        )

        # W&B log (if enabled)
        self._wandb_log({
            "eval/loss": float(val_loss),
            "eval/acc": float(val_acc),
            "eval/auc": float(auc),
            "eval/eer_percent": float(eer),
            "eval/macro_eer_percent": float(macro_eer),
            "eval/eer_threshold": float(eer_thr),
            "eval/min_tDCF": float(min_tdcf) if np.isfinite(min_tdcf) else None,
            "eval/min_tDCF_threshold": float(tdcf_thr) if np.isfinite(min_tdcf) else None,
        })

        # Append a row to metrics.csv tagged as an eval run
        row = {
            "epoch": "eval",
            "train_loss": None,
            "train_acc": None,
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "auc": float(auc) if np.isfinite(auc) else None,
            "eer_percent": float(eer) if np.isfinite(eer) else None,
            "pooled_eer_percent": float(eer) if np.isfinite(eer) else None,
            "macro_eer_percent": float(macro_eer) if np.isfinite(macro_eer) else None,
            "eer_threshold": float(eer_thr) if np.isfinite(eer) else None,
            "min_tDCF": float(min_tdcf) if np.isfinite(min_tdcf) else None,
            "min_tDCF_threshold": float(tdcf_thr) if np.isfinite(min_tdcf) else None,
            "avg_nnz_neighbor_rate": None,
            "avg_grad_norm_projection": None,
            "avg_grad_norm_fuse": None,
            "avg_grad_norm_detection": None,
            "lr_projection": None,
            "lr_fuse": None,
            "lr_detection": None,
            "pos_weight": None,   # You can store the pos_weight you used if you prefer
            "epoch_time_sec": None,
            "top_k": int(getattr(self.config, "top_k", 0)),
            "batch_size": int(getattr(self.config, "eval_batch_size", 0)),
        }
        self._metrics_rows.append(row)
        self.save_metrics_csv()

        return val_loss, val_acc

    def predict(self, audio_path: str, threshold: float = 0.5):
        if (self.vector_db.index is None) or (getattr(self.vector_db.index, "ntotal", 0) == 0):
            logging.warning("Vector DB is empty or not loaded. Retrieval will return zero neighbors.")

        self.radad_model.eval()
        audio_ds = AudioDataset(self.config, is_train=False)

        with torch.no_grad():
            tpp_vec = self.process_audio_batch([audio_path], audio_ds)  # [1, D]

            vecs, lbls, npaths = self.retrieve_similar_vectors(
                tpp_vec, query_paths=[audio_path], exclude_self=True, return_info=True
            )
            if torch.count_nonzero(vecs) == 0:
                vecs, lbls, npaths = self.retrieve_similar_vectors(
                    tpp_vec, query_paths=[audio_path], exclude_self=False, return_info=True
                )

            if vecs.ndim == 2:
                vecs = vecs.unsqueeze(1)
            elif vecs.ndim == 1:
                vecs = vecs.unsqueeze(0).unsqueeze(1)

            use_amp = getattr(self.config, "use_mixed_precision", False) and self.device.type == "cuda"
            if use_amp:
                try:
                    from torch.amp import autocast as _autocast
                    amp_ctx = _autocast("cuda")
                except Exception:
                    from torch.amp import autocast as _autocast
                    amp_ctx = _autocast()
            else:
                class _NullCtx:
                    def __enter__(self): return None
                    def __exit__(self, *args): return False
                amp_ctx = _NullCtx()

            with amp_ctx:
                logits = self.radad_model(vecs, tpp_vec)

            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)

            # logits > 0 ⇒ predict bona-fide (label 1)
            prob = torch.sigmoid(logits).detach().cpu().view(-1).mean().item()
            pred  = "spoof" if prob >= float(threshold) else "bona-fide"
            logit = logits.detach().cpu().view(-1).mean().item()

            retrieved = []
            if isinstance(lbls, torch.Tensor):
                neigh_labels = [int(x) for x in lbls.squeeze(0).detach().cpu().tolist()]
            else:
                neigh_labels = list(lbls[0]) if isinstance(lbls, list) and len(lbls) else []
            neigh_paths = npaths[0] if isinstance(npaths, list) and len(npaths) else []
            for lab, p in zip(neigh_labels, neigh_paths):
                fname = os.path.basename(p) if p else ""
                retrieved.append({"file": fname, "path": p, "label": lab})

            return {
                "prediction": pred,
                "probability_spoof": float(prob),
                "logit": float(logit),
                "retrieved_labels": neigh_labels,
                "retrieved_files": [r["file"] for r in retrieved],
                "retrieved": retrieved
            }

    def validate_no_leakage(self, val_dataset: 'AudioDataset'):
        val_ids = {os.path.basename(item['path']) for item in val_dataset}
        overlap = self.training_file_ids & val_ids
        if overlap:
            raise ValueError(f"Data leakage! {len(overlap)} overlapping files.")
        logging.info("No data leakage detected.")

    # ---------------- single-file save/load ----------------
    def save_models(self, prefix: str = "model"):
        """
        Save a single state_dict for the whole RADADModel.
        File: <data_root>/models/{prefix}_radad.pt
        """
        models_dir = os.path.join(self.config.data_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, f"{prefix}_radad.pt")
        torch.save(self.radad_model.state_dict(), save_path)
        logging.info(f"RADAD model saved to '{save_path}'.")

    def load_models(self, prefix: str = "best_model"):
        """
        Load the single state_dict for the RADADModel.
        File: <data_root>/models/{prefix}_radad.pt
        """
        models_dir = os.path.join(self.config.data_root, "models")
        load_path = os.path.join(models_dir, f"{prefix}_radad.pt")
        state = torch.load(load_path, map_location=self.device)
        self.radad_model.load_state_dict(state)
        self.radad_model.to(self.device)
        logging.info(f"RADAD model loaded from '{load_path}'.")

    def print_split_stats(self,ds, name=""):
        """Print #bonafide and #spoof for a dataset without touching audio."""
        # Prefer the cached labels list if available
        if hasattr(ds, "labels") and isinstance(ds.labels, (list, tuple)):
            labels = np.asarray(ds.labels)
        else:
            # Fallback: iterate quickly via a lightweight DataLoader
            labels_list = []
            tmp_loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
            for batch in tmp_loader:
                y = batch["label"]
                if isinstance(y, torch.Tensor):
                    labels_list.extend(y.cpu().tolist())
                else:
                    labels_list.extend(list(y))
            labels = np.asarray(labels_list)

        total = int(labels.size)
        bona  = int((labels == 1).sum())
        spoof = int((labels == 0).sum())
        ratio_bona  = (bona / total) if total else 0.0
        ratio_spoof = (spoof / total) if total else 0.0
        print(f"{name} set → total: {total}, bonafide: {bona} ({ratio_bona:.2%}), spoof: {spoof} ({ratio_spoof:.2%})")

    def show_curves_inline(
        self,
        read_from_csv: bool = False,
        csv_path: str = None,
        label_val_as: str = "Val",   # change to "Test" if you want the legend to say Test
        smooth: int = 0,             # moving-average window; 0 = no smoothing
        figsize=(12, 5),             # wider default to fit 2 subplots
        ylim_loss=None,
        ylim_acc=None,
    ):
        """
        Display Train vs Val/Test loss and accuracy inline (Colab/Jupyter) in one row (2 columns).
        - If read_from_csv=True (or cache is empty), pulls from metrics.csv.
        - label_val_as lets you name the second curve ("Val" or "Test").
        - smooth>0 applies a moving average with window 'smooth'.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import os

        def _smooth(arr, k):
            if k and k > 1 and len(arr) >= k:
                w = np.ones(k) / float(k)
                return np.convolve(arr, w, mode="valid").tolist()
            return arr

        # Optionally (re)load from metrics.csv if asked or if we have no memory yet
        need_load = read_from_csv or (len(self._train_losses) == 0 and len(self._train_accs) == 0)
        if need_load:
            if csv_path is None:
                csv_path = os.path.join(self.config.data_root, "metrics.csv")
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"metrics.csv not found at: {csv_path}")
            df = pd.read_csv(csv_path)

            train_losses = df["train_loss"].dropna().astype(float).tolist()
            val_losses   = df["val_loss"].dropna().astype(float).tolist() if "val_loss" in df else []
            train_accs   = df["train_acc"].dropna().astype(float).tolist()
            val_accs     = df["val_acc"].dropna().astype(float).tolist() if "val_acc" in df else []
        else:
            train_losses = list(self._train_losses)
            val_losses   = list(self._val_losses)
            train_accs   = list(self._train_accs)
            val_accs     = list(self._val_accs)

        # Epoch axis
        ep_t = list(range(1, len(train_losses) + 1))
        ep_v = list(range(1, len(val_losses) + 1))

        # Smoothing (optional)
        s_train_losses = _smooth(train_losses, smooth)
        s_val_losses   = _smooth(val_losses,   smooth)
        s_train_accs   = _smooth(train_accs,   smooth)
        s_val_accs     = _smooth(val_accs,     smooth)

        # If smoothed, adjust x for 'valid' convolution length
        def _aligned_epochs(ep, arr, k):
            if k and k > 1 and len(ep) >= k:
                shift = k - 1  # 'valid' reduces length by k-1
                return ep[shift:]
            return ep

        ep_t_loss = _aligned_epochs(ep_t, s_train_losses, smooth)
        ep_v_loss = _aligned_epochs(ep_v, s_val_losses,   smooth)
        ep_t_acc  = _aligned_epochs(ep_t, s_train_accs,   smooth)
        ep_v_acc  = _aligned_epochs(ep_v, s_val_accs,     smooth)

        # ---- One figure, two subplots (1 row x 2 cols) ----
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        # LEFT: LOSS
        ax = axes[0]
        if len(s_train_losses):
            ax.plot(ep_t_loss, s_train_losses, label="Train Loss")
        if len(s_val_losses):
            ax.plot(ep_v_loss, s_val_losses,   label=f"{label_val_as} Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Train vs {label_val_as} Loss")
        if ylim_loss is not None:
            ax.set_ylim(ylim_loss)
        ax.legend()
        ax.grid(True, alpha=0.2)

        # RIGHT: ACCURACY
        ax = axes[1]
        if len(s_train_accs):
            ax.plot(ep_t_acc, s_train_accs, label="Train Acc")
        if len(s_val_accs):
            ax.plot(ep_v_acc, s_val_accs,   label=f"{label_val_as} Acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Train vs {label_val_as} Accuracy")
        if ylim_acc is not None:
            ax.set_ylim(ylim_acc)
        ax.legend()
        ax.grid(True, alpha=0.2)

        plt.show()



