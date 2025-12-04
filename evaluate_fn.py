import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
import time
from tqdm import tqdm

# Optional: OULU metrics (https://github.com/.../oulumetrics or similar)
try:
    import oulumetrics
    HAS_OULU = True
except ImportError:
    HAS_OULU = False
    print("[WARN] oulumetrics not installed. APCER/BPCER/ACER will not be computed.")


################################################
############### Eval utils #####################
################################################

def evaluate_all(model, loader, criterion, device, dataset: str = None):
    """
    Generic evaluation function.

    Args:
        model:     trained model
        loader:    DataLoader yielding either:
                     - (inputs, labels), or
                     - (inputs, labels, access_types) [for OULU]
                   where:
                     inputs: (B, T, 3, H, W)
                     labels: (B,) with 0=attack, 1=real
                     access_types (optional): (B,) with OULU attack types:
                        1 - live sample
                        2 - print attack 1
                        3 - print attack 2
                        4 - display attack 1
                        5 - display attack 2
        criterion: loss function (e.g., CrossEntropyLoss)
        device:    torch.device
        dataset:   dataset name string, e.g. 'OULU', 'RA', 'RY', 'RM'

    Returns:
        dict with:
          - test_loss, test_acc
          - auc_roc, eer, hter, far, frr, youdens_index, optimal_threshold
          - avg_inference_time
          - apcer, bpcer, acer  (for OULU, else None)
          - fpr, tpr, labels, probs
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []        # for AUC / EER / etc. -> 0/1 labels
    all_probs = []         # probability for class 1 (real)
    all_times = []         # per-sample inference times
    all_access_types = []  # for OULU APCER/BPCER/ACER (attack types)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            # Support (inputs, labels) OR (inputs, labels, access_types)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    inputs, labels = batch
                    access_types = None
                elif len(batch) == 3:
                    inputs, labels, access_types = batch
                else:
                    raise ValueError(f"Unexpected batch size {len(batch)}. "
                                     f"Expected 2 or 3 elements (inputs, labels[, access_types]).")
            else:
                raise ValueError("Expected batch to be (inputs, labels) or (inputs, labels, access_types).")

            inputs = inputs.to(device, non_blocking=True)   # [B, T, 3, H, W]
            labels = labels.to(device, non_blocking=True).long()  # [B]

            # Start timing for prediction
            start_time = time.time()
            outputs = model(inputs)                        # (B, num_classes)
            inference_time = time.time() - start_time
            # Store per-sample time
            batch_size = inputs.size(0)
            all_times.append(inference_time / max(1, batch_size))

            # Loss calculation
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size

            # Softmax probabilities for positive class (class 1 = "real")
            probs = torch.softmax(outputs, dim=1)[:, 1]  # (B,)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            # Accuracy calculation
            _, predicted = torch.max(outputs, dim=1)     # (B,)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save access types if provided (OULU)
            if access_types is not None:
                all_access_types.extend(
                    access_types.cpu().numpy().tolist()
                )

    # ---------------------------- Classification metrics ----------------------------

    all_labels_np = np.array(all_labels, dtype=np.int32)
    all_probs_np = np.array(all_probs, dtype=np.float32)

    # ROC curve & AUC
    fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
    auc_roc = auc(fpr, tpr)

    # EER
    fnr = 1.0 - tpr
    # index where |FNR - FPR| is minimized
    idx_eer = np.nanargmin(np.absolute(fnr - fpr))
    eer_threshold = thresholds[idx_eer]
    eer = fpr[idx_eer]

    # FAR, FRR, HTER, Youden's Index, Optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    youdens_index = tpr[optimal_idx] - fpr[optimal_idx]
    far = fpr[optimal_idx]
    frr = fnr[optimal_idx]
    hter = (far + frr) / 2.0

    # Average inference time
    avg_inference_time = float(np.mean(all_times)) if len(all_times) > 0 else 0.0

    # Test loss and accuracy
    test_loss = running_loss / max(1, len(loader.dataset))
    test_acc = 100.0 * correct / max(1, total)

    # ---------------------------- OULU-specific metrics ----------------------------

    apcer = bpcer = acer = None
    if dataset == "OULU":
        if not HAS_OULU:
            print("[WARN] oulumetrics not available, skipping APCER/BPCER/ACER.")
        elif len(all_access_types) == 0:
            print("[WARN] No access_type information provided; cannot compute APCER/BPCER/ACER.")
        else:
            # OULU API expects:
            #   y_attack_types: list of attack types (1=live, 2..5=attack types)
            #   y_pred: scores (or binary predictions) for live (1) / attack (0).
            # We give it:
            #   - y_attack_types = access_type from dataset
            #   - y_pred = probabilities for live (class 1)
            y_attack_types = np.array(all_access_types, dtype=np.int32)
            y_pred_scores = all_probs_np  # same as for ROC, prob of class 1 (live)

            # threshold optional; default is 0.5
            apcer, bpcer, acer = oulumetrics.calculate_metrics(
                y_attack_types.tolist(),
                y_pred_scores.tolist()
            )

    # ---------------------------- Return all results ----------------------------

    results = {
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'auc_roc': float(auc_roc),
        'eer': float(eer),
        'hter': float(hter),
        'far': float(far),
        'frr': float(frr),
        'youdens_index': float(youdens_index),
        'optimal_threshold': float(optimal_threshold),
        'avg_inference_time': float(avg_inference_time),
        'fpr': fpr,
        'tpr': tpr,
        'labels': all_labels_np,
        'probs': all_probs_np,
        'apcer': None if apcer is None else float(apcer),
        'bpcer': None if bpcer is None else float(bpcer),
        'acer': None if acer is None else float(acer),
    }

    return results


def generate_evaluation_summary(results):
    # Extract metrics from the results dictionary
    test_loss = results['test_loss']
    test_acc = results['test_acc']
    auc_roc = results['auc_roc']
    eer = results['eer']
    hter = results['hter']
    far = results['far']
    frr = results['frr']
    youdens_index = results['youdens_index']
    optimal_threshold = results['optimal_threshold']
    avg_inference_time = results['avg_inference_time']
    fpr = results['fpr']
    tpr = results['tpr']
    acer = results.get('acer', None)

    print("\n--- Evaluation Summary ---")
    if acer is not None:
        print("HTER (%), AUC-ROC, Test Accuracy (%), ACER (%)")
        print(f"{hter*100:.4f}, {auc_roc:.4f}, {test_acc:.4f}, {acer*100:.4f}")
    else:
        print("HTER (%), AUC-ROC, Test Accuracy (%)")
        print(f"{hter*100:.4f}, {auc_roc:.4f}, {test_acc:.4f}")
    print()

    return hter, auc_roc, test_acc


def plot_roc_curve(results):
    """Generate ROC curve from the evaluation results."""
    fpr = results['fpr']
    tpr = results['tpr']

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {results["auc_roc"]:.4f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_error_histogram(far, frr, eer):
    """Plot histograms of FAR, FRR, and EER values."""
    plt.figure(figsize=(10, 6))

    # Plot FAR
    plt.subplot(1, 3, 1)
    plt.bar(['FAR'], [far], color='red')
    plt.ylabel('Rate')
    plt.title('False Acceptance Rate (FAR)')

    # Plot FRR
    plt.subplot(1, 3, 2)
    plt.bar(['FRR'], [frr], color='blue')
    plt.ylabel('Rate')
    plt.title('False Rejection Rate (FRR)')

    # Plot EER
    plt.subplot(1, 3, 3)
    plt.bar(['EER'], [eer], color='green')
    plt.ylabel('Rate')
    plt.title('Equal Error Rate (EER)')

    plt.tight_layout()
    plt.show()


def plot_inference_time(avg_inference_time):
    """Plot inference time as a bar chart."""
    plt.figure(figsize=(6, 4))
    plt.bar(['Average Inference Time'], [avg_inference_time], color='purple')
    plt.ylabel('Time (seconds)')
    plt.title('Average Inference Time per Sample')
    plt.show()
