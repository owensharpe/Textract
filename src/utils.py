"""
File: utils.py
Author:
Description: some helper functions for the model during training and inference
"""

# import libraries
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
import os


# plotting training curves from the TensorBoard logs
def plot_curves(log_dir, save_path=None):
    """
    :param log_dir: directory where the TensorBoard logs are
    :param save_path: specified filepath to save plots
    :return: Null (plotting)
    """

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("Please install tensorboard to plot training curves")
        return

    # load events
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # extract data
    train_loss = [(s.step, s.value) for s in event_acc.Scalars('Loss/train')]
    val_loss = [(s.step, s.value) for s in event_acc.Scalars('Loss/val')]
    val_bleu = [(s.step, s.value) for s in event_acc.Scalars('BLEU/val')]

    # build plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # loss plot
    ax1.plot(*zip(*train_loss), label='Train Loss', color='blue')
    ax1.plot(*zip(*val_loss), label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # bleu plot
    ax2.plot(*zip(*val_bleu), label='Val BLEU', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BLEU Score')
    ax2.set_title('Validation BLEU Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# visualize the model predictions with attention
def viz_predictions(model, dataloader, vocab, n_samples=5, save_dir=None):
    """
    :param model: specified model
    :param dataloader: specified dataloader
    :param vocab: the model vocabulary
    :param n_samples: number of samples within data
    :param save_dir: specified save directory for plots (if passed)
    :return: Null (plotting)
    """

    device = next(model.parameters()).device
    model.eval()

    batch = next(iter(dataloader))
    images = batch['images'][:n_samples].to(device)
    targets = batch['latex_texts'][:n_samples]

    with torch.no_grad():

        # retrieve encoder features
        encoder_out = model.encoder(images)

        # generate predictions using encoder features
        pred = model.decoder.generate(encoder_out)

        # retrieve attention weights
        targ_tokens = batch['latex'][:n_samples].to(device)
        captions = targ_tokens[:, :-1]
        outputs, att_weights = model.decoder(encoder_out, captions)

    # build visualizations
    for i in range(n_samples):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f'Sample {i + 1}', fontsize=16)

        # show original image
        img = images[i].cpu()
        if img.shape[0] == 1:  # grayscale
            img = img.squeeze(0)
            axes[0, 0].imshow(img, cmap='gray')
        else:  # or RGB
            img = img.permute(1, 2, 0)
            axes[0, 0].imshow(img)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        # show attention for first 8 tokens (this can be changed; 8 is clean in a grid format)
        attention = att_weights[i].cpu().numpy()
        pred_tokens = vocab.decode(pred[i].cpu().tolist())
        pred_token_list = vocab.tokenize(pred_tokens)

        for j in range(1, min(9, len(pred_token_list))):
            row = (j - 1) // 4
            col = (j - 1) % 4 + 1

            if j - 1 < attention.shape[0]:
                att_map = attention[j - 1].reshape(7, 7)
                axes[row, col].imshow(att_map, cmap='hot', interpolation='bilinear')
                axes[row, col].set_title(f'{pred_token_list[j - 1]}')
            axes[row, col].axis('off')

        # add text with predictions
        pred_latex = vocab.decode(pred[i].cpu().tolist())
        plt.figtext(0.1, 0.02, f'Target: {targets[i]}', fontsize=10, ha='left')
        plt.figtext(0.1, 0.01, f'Predicted: {pred_latex}', fontsize=10, ha='left')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'prediction_{i + 1}.png'))
        else:
            plt.show()
        plt.close()


# calculate accuracy of sequences (matching exactly)
def calc_sequence_accuracy(pred, targ, pad_idx=0):
    """
    :param pred: prediction labels
    :param targ: target labels
    :param pad_idx: padding index
    :return: exact match accuracy
    """
    batch_size = pred.shape[0]
    correct = 0

    for i in range(batch_size):
        pred_seq = pred[i]
        targ_seq = targ[i]

        # remove padding
        pred_seq = pred_seq[pred_seq != pad_idx]
        targ_seq = targ_seq[targ_seq != pad_idx]

        # check if sequences match exactly
        if pred_seq.shape[0] == targ_seq.shape[0] and torch.all(pred_seq == targ_seq):
            correct += 1

    return correct / batch_size


# grab model architecture summary
def retrieve_model_summary(model, input_shape=(1, 3, 224, 224)):
    """
    :param model: specified model
    :param input_shape: dimension of input data
    :return: Null (printing)
    """

    device = next(model.parameters()).device
    summary(model.encoder, input_shape[1:], device=str(device))

    print("\nTotal model parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"Non-trainable: {total_params - trainable_params:,}")


def main():
    print('')


if __name__ == '__main__':
    main()
