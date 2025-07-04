import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
import pandas as pd


def plot_learning_curves(history, save_path=None, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path)
        plt.close()
    else:
        return fig


def plot_depth_comparison(results, save_path=None, reg_enabled=False):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for res in results:
        label = f"Depth {res['depth']}"
        if res['reg']:
            label += " (reg)"
        axes[0].plot(res['history']['test_accs'], label=label)
    axes[0].set_title('Test Accuracy Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    depths = [r['depth'] for r in results]
    gaps = [r['overfit_gap'] for r in results]
    axes[1].bar(depths, gaps)
    axes[1].set_title('Overfitting Gap by Depth')
    axes[1].set_xlabel('Depth')
    axes[1].set_ylabel('Train-Test Accuracy Gap')

    times = [r['training_time'] for r in results]
    axes[2].bar(depths, times)
    axes[2].set_title('Training Time by Depth')
    axes[2].set_xlabel('Depth')
    axes[2].set_ylabel('Time (seconds)')

    plt.suptitle(f"Depth Comparison {'with Regularization' if reg_enabled else ''}")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path)
        plt.close()
    else:
        return fig


def plot_width_comparison(experiments, plots_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for exp in experiments:
        plt.plot(exp['history']['test_accs'],
                 label=f"{exp['profile']} ({exp['widths']})")
    plt.title('Test Accuracy by Width Profile')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar([exp['profile'] for exp in experiments],
            [exp['training_time'] for exp in experiments])
    plt.title('Training Time by Width Profile')
    plt.ylabel('Time (seconds)')

    plt.tight_layout()
    plt.savefig(plots_dir / 'width_comparison.png')
    plt.close()

    df = pd.DataFrame([{
        'Profile': exp['profile'],
        'Widths': str(exp['widths']),
        'Parameters': exp['parameters'],
        'Max Test Acc': exp['max_test_acc'],
        'Time (s)': exp['training_time']
    } for exp in experiments])
    print("\nWidth Experiment Results:")
    print(df.to_markdown(index=False))


def save_metrics_to_json(metrics, filepath):
    Path(filepath).parent.mkdir(exist_ok=True, parents=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)