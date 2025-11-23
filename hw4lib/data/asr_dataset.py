from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

class ASRDataset(Dataset):
    def __init__(
            self,
            partition:Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config:dict,
            tokenizer:H4Tokenizer,
            isTrainPartition:bool,
            global_stats:Optional[Tuple[torch.Tensor, torch.Tensor]]=None
    ):
        """
        Initialize the ASRDataset for ASR training/validation/testing.
        """
        # Store basic configuration
        self.config          = config
        self.partition       = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer       = tokenizer

        # --- Special tokens (using properties you implemented in H4Tokenizer) ---
        # TODO: Get tokenizer ids for special tokens (eos, sos, pad)
        
        self.eos_token = self.tokenizer.eos_id
        self.sos_token = self.tokenizer.sos_id
        self.pad_token = self.tokenizer.pad_id


        # --- Data paths ---
        self.fbank_dir = os.path.join(self.config["root"], self.partition, "fbank")

        # IMPORTANT: store **filenames**, not full paths
        self.fbank_files = sorted(
            [f for f in os.listdir(self.fbank_dir) if f.endswith(".npy")]
        )

        # subset handling (same style as LM dataset)
        # Always keep the full test set so recognition writes all rows (e.g., 2620)
        subset = self.config.get("subset", 1.0)
        if self.partition == "test-clean" and subset < 1.0:
            subset = 1.0

        if subset < 1.0:
            subset_size = max(1, int(len(self.fbank_files) * subset))
            self.fbank_files = self.fbank_files[:subset_size]

        self.length = len(self.fbank_files)

        if self.partition != "test-clean":
            self.text_dir = os.path.join(self.config["root"], self.partition, "text")

            # again: filenames only
            self.text_files = sorted(
                [f for f in os.listdir(self.text_dir) if f.endswith(".npy")]
            )

            if subset < 1.0:
                self.text_files = self.text_files[:self.length]

            # Verify data alignment (same basename before ".npy")
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        # Initialize lists to store features and transcripts
        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []

        # DO NOT MODIFY counters
        self.total_chars  = 0
        self.total_tokens = 0
        self.feat_max_len = 0
        self.text_max_len = 0

        # Welford accumulators for global MVN
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions when using global_mvn")
            count = 0
            mean  = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2    = torch.zeros(self.config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # --- Load features (num_feats, time) ---
            feat_path = os.path.join(self.fbank_dir, self.fbank_files[i])
            feat = np.load(feat_path)          # (F, T)
            feat = feat[: self.config['num_feats'], :].astype(np.float32)

            self.feats.append(feat)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            # update global stats if needed
            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.FloatTensor(feat)      # (F, T)
                batch_count = feat_tensor.shape[1]
                count += batch_count

                delta  = feat_tensor - mean.unsqueeze(1)
                mean  += delta.mean(dim=1)
                delta2 = feat_tensor - mean.unsqueeze(1)
                M2    += (delta * delta2).sum(dim=1)

            # --- transcripts (non-test partitions only) ---
            if self.partition != "test-clean":
                text_path = os.path.join(self.text_dir, self.text_files[i])
                arr = np.load(text_path, allow_pickle=True)
                transcript = "".join(arr.tolist())

                self.total_chars += len(transcript)

                tokenized = self.tokenizer.encode(transcript)
                self.total_tokens += len(tokenized)
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                self.transcripts_shifted.append([self.sos_token] + tokenized)
                self.transcripts_golden.append(tokenized + [self.eos_token])

        # avg chars per token
        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        )

        if self.partition != "test-clean":
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # final global stats
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                variance = M2 / (count - 1)
                self.global_std  = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # SpecAugment transforms (only used if specaug=True)
        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        return self.avg_chars_per_token

    def __len__(self) -> int:
        # previously was NotImplementedError
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = torch.FloatTensor(self.feats[idx])  # (F, T)

        # normalization
        if self.config['norm'] == 'global_mvn':
            assert hasattr(self, "global_mean") and hasattr(self, "global_std")
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)
        elif self.config['norm'] == 'none':
            pass

        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
            golden_transcript  = torch.LongTensor(self.transcripts_golden[idx])

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # features: list of (F, T) -> (T, F)
        batch_feats  = [feat.transpose(0, 1) for feat, _, _ in batch]   # list of (T, F)
        feat_lengths = torch.LongTensor([f.shape[0] for f in batch_feats])

        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=0.0)  # (B, T, F)

        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            batch_shifted = [shifted for _, shifted, _ in batch]
            batch_golden  = [golden  for _, _, golden in batch]

            transcript_lengths = torch.LongTensor([t.size(0) for t in batch_shifted])

            padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)
            padded_golden  = pad_sequence(batch_golden,  batch_first=True, padding_value=self.pad_token)

        # SpecAugment (train only)
        if self.config["specaug"] and self.isTrainPartition:
            padded_feats = padded_feats.permute(0, 2, 1)   # (B, F, T)

            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)

            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)

            padded_feats = padded_feats.permute(0, 2, 1)   # back to (B, T, F)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
