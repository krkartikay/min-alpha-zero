import numpy as np
import torch
from tqdm import tqdm


def measure_min_achievable_loss(dataset, eps=1e-12):
    """
    Computes the dataset-wide minimal achievable loss under your objective:
      loss = MSE(value_pred, final_value) + KL(target_policy || policy_pred)
    assuming the model can output a separate optimum for each unique position.

    Returns: dict with breakdown and totals.
    """
    # Per-position accumulators keyed by the exact board tensor bytes
    groups = {}  # key -> {'n', 'sum_y', 'sum_y2', 'sum_q', 'sum_q_log_q'}
    total_samples = 0

    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        item = dataset[i]
        board = item["board_tensor"]  # (7,8,8), torch or np
        counts = item["child_visit_counts"]  # (4096), torch or np
        value = float(item["final_value"])  # scalar

        # Convert to numpy
        if torch.is_tensor(board):
            board_np = board.cpu().numpy()
        else:
            board_np = np.asarray(board)
        if torch.is_tensor(counts):
            counts_np = counts.float().cpu().numpy()
        else:
            counts_np = np.asarray(counts, dtype=np.float32)

        # Use raw board tensor bytes as a key for "same position"
        key = board_np.tobytes(order="C")

        # Normalize counts -> target policy q
        s = counts_np.sum()
        if s <= 0:
            # If a row has zero visits, treat as uniform over nonzero legal moves (or skip).
            # Here we skip its policy contribution but keep value; you can choose otherwise.
            q = None
        else:
            q = counts_np / s
            # clamp for log
            q_clip = np.clip(q, eps, 1.0)
            q_log_q_sum = float((q * np.log(q_clip)).sum())

        # Init bucket
        g = groups.get(key)
        if g is None:
            g = {
                "n": 0,
                "sum_y": 0.0,
                "sum_y2": 0.0,
                "sum_q": None,  # np.array(4096,)
                "sum_q_log_q": 0.0,  # sum over samples of q · log q
                "n_policy": 0,  # count of samples contributing to policy stats
            }
            groups[key] = g

        # Update value stats
        g["n"] += 1
        g["sum_y"] += value
        g["sum_y2"] += value * value
        total_samples += 1

        # Update policy stats if we have a valid q
        if q is not None:
            if g["sum_q"] is None:
                g["sum_q"] = q.astype(np.float64, copy=True)
            else:
                g["sum_q"] += q
            g["sum_q_log_q"] += q_log_q_sum
            g["n_policy"] += 1

    # Aggregate minimal losses
    total_min_value_mse = 0.0
    total_min_policy_kl = 0.0
    total_policy_samples = 0

    for g in tqdm(groups.values(), desc="Aggregating groups"):
        n = g["n"]
        # Value: per-sample minimal MSE = Var(y) = E[y^2] - (E[y])^2
        Ey = g["sum_y"] / n
        Ey2 = g["sum_y2"] / n
        var = max(0.0, Ey2 - Ey * Ey)  # numerical safety
        total_min_value_mse += n * var

        # Policy: per-sample minimal KL = E[ KL(q || mean_q) ]
        m = g["n_policy"]
        if m > 0 and g["sum_q"] is not None:
            mean_q = g["sum_q"] / m
            mean_q_clip = np.clip(mean_q, eps, 1.0)
            # E[q · log q]  = (1/m) * sum_i q_i · log q_i  (we stored the sum)
            Eq_log_q = g["sum_q_log_q"] / m
            # E[q · log mean_q] = (1/m) * (sum_i q_i) · log mean_q = (sum_q/m) · log mean_q
            Eq_log_mean = float((mean_q * np.log(mean_q_clip)).sum())
            kl_min_group = (
                Eq_log_q - Eq_log_mean
            )  # this is already averaged per sample in the group
            total_min_policy_kl += m * kl_min_group
            total_policy_samples += m

    # Convert totals back to dataset-average losses
    avg_min_value_mse = total_min_value_mse / max(1, total_samples)
    avg_min_policy_kl = total_min_policy_kl / max(1, total_policy_samples)
    avg_min_total = avg_min_value_mse + avg_min_policy_kl

    return {
        "num_positions": len(groups),
        "num_samples_total": total_samples,
        "num_policy_samples": total_policy_samples,
        "avg_min_value_mse": float(avg_min_value_mse),
        "avg_min_policy_kl": float(avg_min_policy_kl),
        "avg_min_total_loss": float(avg_min_total),
    }


# Example usage:
from dataset import TrainingDataset

ds = TrainingDataset("training_data.bin")
print(measure_min_achievable_loss(ds))
