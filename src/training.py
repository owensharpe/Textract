"""
File: training.py
Author:
Description: training the image-to-latex model
"""

# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import json
import argparse
from torchvision import transforms

# import modules
from dataset import create_dataloaders
from models import ImageToLatex

# evaluation libraries
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# small helper class to help keep track of averages
class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# calculate the loss of an epoch
def calculate_loss(outputs, targets, pad_idx=0):
    """
    :param outputs: the outputs from the model
    :param targets: the target labels
    :param pad_idx: padding token index
    :return: the cross entropy loss for the classification
    """

    # reshape for loss calculation
    batch_size, max_seq, vocab_size = outputs.shape

    # flatten outputs and targets for faster computation
    outputs = outputs.reshape(-1, vocab_size)
    targets = targets.reshape(-1)

    # calculating the loss ignoring the paddings
    cel = nn.CrossEntropyLoss(ignore_index=pad_idx)
    loss = cel(outputs, targets)

    return loss


# calculate token-level accuracy of an epoch
def calculate_accuracy(outputs, targets, pad_idx=0):
    """
    :param outputs: the outputted tokens from the model
    :param targets: the target labels
    :param pad_idx: padding token idx
    :return: the model's accuracy
    """

    # retrieve predictions
    _, pred = outputs.max(dim=-1)

    # non-padding token mask
    mask = targets != pad_idx

    # calculate accuracy on non-padding tokens
    correct = (pred == targets) & mask
    acc = correct.sum().float() / mask.sum().float()

    return acc.item()


# calculate the BLEU score for a given batch
def calculate_bleu(pred_ids, target_ids, pad_idx=0, end_idx=2):
    """
    :param pred_ids: the ids of the predicted values
    :param target_ids: the ids of the target values
    :param pad_idx: padding token index
    :param end_idx: end of sequence index
    :return: BLEU score
    """
    bleu_scores = []
    smoothing = SmoothingFunction().method1

    for pred, target in zip(pred_ids, target_ids):

        # convert to tokens (removing special ones)
        pred_tokens = []
        for i in pred:
            if i == end_idx:
                break
            if i not in [pad_idx, 1, 2]:  # skip command tokens
                pred_tokens.append(str(i))

        targ_tokens = []
        for i in target:
            if i == end_idx:
                break
            if i not in [pad_idx, 1, 2]:  # skip command tokens
                targ_tokens.append(str(i))

        if len(pred_tokens) == 0:
            pred_tokens = ['0']
        if len(targ_tokens) == 0:
            targ_tokens = ['0']

        # calculate BLEU score
        bleu_score = sentence_bleu([targ_tokens], pred_tokens,
                              smoothing_function=smoothing, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(bleu_score)

    return np.mean(bleu_scores)


# train a specific epoch
def train_epoch(model, dataloader, optimizer, vocab, device, grad_clip=5.0, accumulation_steps=1):
    """
    :param model: the provided model
    :param dataloader: the provided data loader
    :param optimizer: the provided optimizer function
    :param vocab: the provided latex vocabulary
    :param device: the provided device
    :param grad_clip: gradient clipping value
    :param accumulation_steps: number of accumulation steps
    :return: the loss and accuracy average
    """

    model.train()

    # clear gradients just in case
    optimizer.zero_grad()

    # calculate metrics
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # using tqdm for progress bar
    progress = tqdm(dataloader, desc='Training')

    for i, batch in enumerate(progress):

        # move to device your computer is using
        images = batch['images'].to(device)
        latex_tokens = batch['latex'].to(device)

        # input=all tokens except last; target=all tokens except first
        input_tok = latex_tokens[:, :-1]
        target_tok = latex_tokens[:, 1:]

        # run a forward pass
        outputs, att_weights = model(images, input_tok)

        # calculate loss
        loss = calculate_loss(outputs, target_tok, pad_idx=vocab.token_to_idx['<PAD>'])
        loss = loss / accumulation_steps

        # backward pass (backpropagation)
        loss.backward()

        # gradient accumulation
        if (i+1) % accumulation_steps == 0:

            # clip gradients
            clip_grad_norm_(model.parameters(), grad_clip)

            # update weights
            optimizer.step()
            optimizer.zero_grad()

        # calculate accuracy at batch
        acc = calculate_accuracy(outputs, target_tok, pad_idx=vocab.token_to_idx['<PAD>'])

        # update metrics
        loss_meter.update(loss.item() * accumulation_steps, images.size(0))
        acc_meter.update(acc, images.size(0))

        # update the progress bar
        progress.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'accuracy': f'{acc_meter.avg:.4f}'
        })

    return loss_meter.avg, acc_meter.avg


# validate the model
def validate(model, dataloader, vocab, device):
    """
    :param model: the provided model
    :param dataloader: the provided dataloader
    :param vocab: the provided latex vocabulary
    :param device: the provided device
    :return: validation loss, accuracy, and BLEU score
    """

    model.eval()

    # calculate metrics
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    bleu_meter = AverageMeter()

    # using tqdm for progress bar
    progress = tqdm(dataloader, desc='Validation')

    with torch.no_grad():
        for batch in progress:

            # move to device
            images = batch['images'].to(device)
            latex_tokens = batch['latex'].to(device)

            # input=all tokens except last; target=all tokens except first
            input_tok = latex_tokens[:, :-1]
            target_tok = latex_tokens[:, 1:]

            # run a forward pass
            outputs, _ = model(images, input_tok)

            # calculate loss
            loss = calculate_loss(outputs, target_tok, pad_idx=vocab.token_to_idx['<PAD>'])

            # calculate accuracy
            accuracy = calculate_accuracy(outputs, target_tok, pad_idx=vocab.token_to_idx['<PAD>'])

            # calculate bleu scores
            pred = model(images)
            bleu_score = calculate_bleu(pred.cpu().numpy(), latex_tokens.cpu().numpy())

            # update meters
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy, images.size(0))
            bleu_meter.update(bleu_score, images.size(0))

            # update progress bar
            # update the progress bar
            progress.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'accuracy': f'{acc_meter.avg:.4f}',
                'BLEU score': f'{bleu_meter.avg:.4f}'
            })

    return loss_meter.avg, acc_meter.avg, bleu_meter.avg


# create a save checkpoint for model training
def save_model_checkpoint(state, filename='model_checkpoint.pth'):
    """
    :param state: state of current model
    :param filename: specified filename for the checkpoint
    :return: Null (saving file)
    """

    torch.save(state, filename)
    print(f"Current checkpoint saved to {filename}")


# load a given checkpoint from a file
def load_model_checkpoint(filename, model, optimizer=None):
    """
    :param filename: specified filename for the checkpoint
    :param model: provided model
    :param optimizer: provided optimizer
    :return: the epoch and best validation loss left off on
    """

    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['best_val_loss']


# actually train the model
def train(args):
    """
    :param args: specified arguments
    :return:
    """

    # set training device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.9, 0.9, 0.9], std=[0.1, 0.1, 0.1])  # Adjusted for 3 channels
    ])

    # create dataloaders with the transform
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform
    )

    # save latex vocabulary
    vocab.save(os.path.join(args.output_dir, 'vocab.json'))

    # create model
    print("Creating model...")
    model = ImageToLatex(
        vocab_size=len(vocab.token_to_idx),
        encoder_hidden_d=args.encoder_d,
        decoder_hidden_d=args.decoder_d,
        embed_d=args.embed_d,
        attention_d=args.attention_d,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pretrained_encoder=args.pretrained
    ).to(device)

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {train_params}")

    # create optimizer and scheduler (we're going to use adam)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) #verbose=True isn't a valid arg

    # also using a TensorBoard writer
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # our training variables
    starting_epoch = 0
    best_val_loss= float('inf')
    patience_count = 0

    # load checkpoint if we're picking up from somewhere
    if args.resume:
        starting_epoch, best_val_loss = load_model_checkpoint(args.resume, model, optimizer)
        print(f"Picking up from epoch {starting_epoch}")

    # running training loop
    print(f"Starting Training...")
    for epoch in range(starting_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")  # just making space between epochs
        print(f"{'='*50}")

        # train model
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, vocab, device,
            grad_clip=args.grad_clip, accumulation_steps=args.accumulation_steps
        )

        # validate model
        val_loss, val_acc, val_bleu = validate(model, val_loader, vocab, device)

        # update the scheduler
        scheduler.step(val_loss)

        # update logs
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
        tb_writer.add_scalar('BLEU/val', val_bleu, epoch)
        tb_writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # print current status of epochs
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val BLEU: {val_bleu:.4f}")

        # save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1

        # save an updated checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_bleu': val_bleu,
            'best_val_loss': best_val_loss,
            'vocab': vocab.token_to_idx
        }

        save_model_checkpoint(checkpoint, os.path.join(args.output_dir, 'last_model_checkpoint.pth'))

        # in case we stop early
        if patience_count >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    # final eval
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    test_loss, test_acc, test_bleu = validate(model, test_loader, vocab, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test BLEU: {test_bleu:.4f}")

    tb_writer.close()
    print("\nTraining completed!")


def main():

    parser = argparse.ArgumentParser(description='Train Image-to-LaTeX model')

    # data arguments
    parser.add_argument('--data-dir', type=str, default='crohme_images',
                        help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='experiments/exp1',
                        help='Path to save outputs')

    # model arguments
    parser.add_argument('--encoder-d', type=int, default=256,
                        help='Encoder hidden dimension')
    parser.add_argument('--decoder-d', type=int, default=256,
                        help='Decoder hidden dimension')
    parser.add_argument('--embed-d', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--attention-d', type=int, default=256,
                        help='Attention dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained encoder')

    # training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=5.0,
                        help='Gradient clipping value')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # other arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # set random seeds for running
    torch.manual_seed(args.seed)

    # create an output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # train
    train(args)


if __name__ == '__main__':
    main()

# TO-DO: FIX DATA DIRECTORY FILE MANAGEMENT
