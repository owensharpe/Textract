# import torch
# from PIL import Image
# from torchvision import transforms
# from train_cnn import LatexCNN
# from dataset import create_dataloaders
# import os
# import random

# def predict_latex(model, image_path, vocab, transform, device, max_length=50):

#     image = Image.open(image_path)
#     image = transform(image).unsqueeze(0).to(device)
#     pred_caption = [vocab.token_to_idx['<SOS>']]
    
#     with torch.no_grad():
#         for x in range(max_length):
#             caption_tensor = torch.LongTensor(pred_caption).unsqueeze(0).to(device)
#             output = model(image, caption_tensor)
#             next_token_idx = output[0, -1].argmax().item()
            
#             if next_token_idx == vocab.token_to_idx['<EOS>']:
#                 break
                
#             pred_caption.append(next_token_idx)
    
#     # convert indices to tokens
#     predicted_tokens = [vocab.idx_to_token[idx] for idx in pred_caption[1:]]  # skip <SOS>
#     return ' '.join(predicted_tokens)

# def main():
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     base_dir = os.path.dirname(script_dir)
#     data_dir = os.path.join(base_dir, 'data', 'crohme_images', 'TC11_CROHME23', 'INKML', 'test')
    
#     #same transforms as training
#     transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.9], std=[0.1])
#     ])
    
#     # load vocab from dataset
#     _, _, _, vocab = create_dataloaders(
#         data_dir,
#         batch_size=1,
#         transform=transform
#     )
    
#     # load the model
#     if not os.path.exists('model.pth'):
#         print("Error: model.pth not found. Please train the model first using train_cnn.py")
#         return
        
#     model = LatexCNN(len(vocab.token_to_idx)).to(device)
#     model.load_state_dict(torch.load('model.pth'))
#     model.eval()
#     print("Model loaded successfully")

#     available_files = {}
#     for root, _, files in os.walk(data_dir):
#         for file in files:
#             available_files[file] = os.path.join(root, file)

#     if(len(available_files) > 0):
#         image_path = random.choice(list(available_files.values()))
#         print(f"Testing with image: {image_path}")
#         predicted_latex = predict_latex(model, image_path, transform, vocab, device)
#         print(f"Predicted LaTeX: {predicted_latex}")
#     else:
#         print("No test images found in the dataset")

# if __name__ == '__main__':
#     main() 


"""
File: inference.py
Author:
Description: Inference script for the model to make predictions on unseen data
"""

# import libraries
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# import modules
from dataset import LaTeXVocab
from dataset import create_dataloaders
from models import ImageToLatex


# load the model
def load_model(checkpoint_path, device='cpu'):
    """
    :param checkpoint_path: specified filepath for model
    :param device: device that will be used
    :return: the model with its vocabulary
    """

    # load the model from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # load model vocabulary
    vocab = LaTeXVocab(min_freq=2)
    vocab.token_to_idx = checkpoint['vocab']
    vocab.idx_to_token = {int(idx): tok for tok, idx in vocab.token_to_idx.items()}

    # now create model
    model = ImageToLatex(
        vocab_size=len(vocab.token_to_idx),
        encoder_hidden_d=256, 
        decoder_hidden_d=256, #THESE TWO NEED TO MATCH
        embed_d=256,
        attention_d=256,
        num_layers=2,
        dropout=0.0
    ).to(device)

    # load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, vocab


# preprocessing a given image
def preprocess_image(image_path, transform=None):
    """
    :param image_path: specified filepath for image
    :param transform: applied transformations for image
    :return: RGB color in tensor form
    """

    if transform is None:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.9], std=[0.1])
        ])

    # convert the grayscale to RGB
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image)

    return img_tensor.unsqueeze(0)


# decoding a sequence with beam search (should give us improved results)
def beam_search_decode(model, image, vocab, beam_width=5, max_length=150):
    """
    :param model: given model
    :param image: given preprocessed image tensor
    :param vocab: model vocabulary
    :param beam_width: number of beams
    :param max_length: maximum sequence length
    :return: best sequence with score
    """

    # get device and specify start and end token
    device = next(model.parameters()).device
    start_tok = vocab.token_to_idx['< SOS >']
    end_tok = vocab.token_to_idx['<EOS>']

    # encode the image
    with torch.no_grad():
        encoder_out = model.encoder(image.to(device))

    # initialize beams
    beams = [(torch.LongTensor([start_tok]).to(device), 0.0, None, None, False)]

    # get initial hidden states
    h, c = model.decoder.init_hidden_state(encoder_out)
    beams[0] = (beams[0][0], 0.0, h, c, False)

    # flatten encoder output for attention
    batch_size = encoder_out.size(0)
    encoder_d = encoder_out.size(1)
    encoder_out_flat = encoder_out.view(batch_size, encoder_d, -1)
    encoder_out_flat = encoder_out_flat.permute(0, 2, 1)

    for step in range(max_length):
        candidates = []

        for seq, score, h, c, finished in beams:
            if finished:
                candidates.append((seq, score, h, c, finished))
                continue

            # find last token
            last_tok = seq[-1].unsqueeze(0)

            # now embed token
            with torch.no_grad():
                embed = model.decoder.embedding(last_tok)
                embed = model.decoder.dropout(embed)

                # attention
                weights, context = model.decoder.attention(encoder_out_flat, h[-1])

                # lstm input
                lstm_input = torch.cat([embed, context], dim=1).unsqueeze(1)

                # lstm forward pass
                lstm_out, (updated_h, updated_c) = model.decoder.lstm(lstm_input, (h, c))

                # retrieve probabilities
                output = model.decoder.out_proj(lstm_out.squeeze(1))
                log_probs = F.log_softmax(output, dim=-1)

            # get the top k tokens
            top_k_log_prob, top_k_indices = log_probs.topk(beam_width)

            for i in range(beam_width):
                token = top_k_indices[0, i]
                tok_score = top_k_log_prob[0, i].item()

                updated_seq = torch.cat([seq, token.unsqueeze(0)])
                updated_score = tok_score + score

                if token == end_tok:
                    candidates.append((updated_seq, updated_score, updated_h, updated_c, True))
                else:
                    candidates.append((updated_seq, updated_score, updated_h, updated_c, False))

        # select top beams
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

        # stop if all beams have been terminated
        if all(finished for _, _, _, _, finished, in beams):
            break

    # return best sequence based on score
    best_seq, best_score, _, _, _ = beams[0]
    return best_seq.cpu().tolist(), best_score


# visualizing attention during generation
def attention_viz(model, image, latex_seq, vocab, save_path=None):
    """
    :param model: specified model
    :param image: specified image
    :param latex_seq: given latex sequence
    :param vocab: the model vocabulary
    :param save_path: if we're going to save visualization, specify path
    :return: Null (plotting)
    """

    device = next(model.parameters()).device

    # generate sequence with attention training
    with torch.no_grad():
        encoder_out = model.encoder(image.to(device))
        _, att_weights = model.decoder(encoder_out, latex_seq.to(device))

    # attention weights to numpy
    attention = att_weights.cpu().numpy()[0]

    # create plot
    fig, axes = plt.subplots(2, 5, figsize=(15,6))
    axes = axes.flatten()

    # decode tokens for labels
    tokens = [vocab.idx_to_token.get(idx, '<UNK>') for idx in latex_seq[0].tolist()]

    # plot attention of first 10 tokens
    for i in range(min(10, len(tokens))):
        att_map = attention[i].reshape(7, 7)
        axes[i].imshow(att_map, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'Token: {tokens[i]}')
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# predicting a single inputted image
def predict_image(model, vocab, image_path, beam_width=5):
    """
    :param model: specified model
    :param vocab: the model vocabulary
    :param image_path: specified file path to image
    :param beam_width: number of sequences to maintain during beam search
    :return: the latex, the sequence, and the score
    """

    # preprocess image
    device = next(model.parameters()).device
    image = preprocess_image(image_path)

    # generate prediction
    if beam_width > 1:
        seq, score = beam_search_decode(model, image, vocab, beam_width=beam_width)
    else:  # doing greedy decoding instead
        with torch.no_grad():
            seq = model(image.to(device))
        seq = seq[0].cpu().tolist()
        score = 0.0

    # decode to latex
    latex = vocab.decode(seq)

    return latex, seq, score


# evaluate model on a given test set
def eval_test_set(model, vocab, test_loader, n_samples=10):
    """
    :param model: specified model
    :param vocab: the model vocabulary
    :param test_loader: test partition dataloader
    :param n_samples: number of samples to use from dataset
    :return: Null (printing)
    """

    device = next(model.parameters()).device

    # grab a batch
    batch = next(iter(test_loader))
    images = batch['images'].to(device)
    targ_latex = batch['latex_texts']

    # make predictions
    with torch.no_grad():
        pred = model(images)

    # show some examples from predictions
    for i in range(min(len(images), n_samples)):
        pred_seq = pred[i].cpu().tolist()
        pred_latex = vocab.decode(pred_seq)

        print(f"\nExample {i + 1}:")
        print(f"Target: {targ_latex[i]}")
        print(f"Predicted: {pred_latex}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description='Image-to-LaTeX Model Inference')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for inference')
    parser.add_argument('--beam-width', type=int, default=5,
                        help='Beam width for beam search')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize attention weights')
    parser.add_argument('--test', action='store_true',
                        help='Run on test set')
    parser.add_argument('--data-dir', type=str, default='crohme_images',
                        help='Path to data directory (for test set)')

    args = parser.parse_args()

    # specify device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model
    print("Loading model...")
    model, vocab = load_model(args.checkpoint, device)

    if args.image:  # if single image
        print(f"\nProcessing image: {args.image}")
        latex, sequence, score = predict_image(
            model, vocab, args.image, beam_width=args.beam_width
        )

        print(f"\nPredicted LaTeX: {latex}")
        print(f"Confidence score: {score:.4f}")

        if args.visualize:  # prepare sequence tensor for visualization (if we are visualizing)
            seq_tensor = torch.LongTensor([sequence]).to(device)
            image_tensor = preprocess_image(args.image).to(device)
            attention_viz(model, image_tensor, seq_tensor, vocab)

    elif args.test:  # or if it's a whole data partition
        print("\nLoading test data...")

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.9, 0.9, 0.9], std=[0.1, 0.1, 0.1])  # Adjusted for 3 channels
        ])

        _, _, test_loader, _ = create_dataloaders(
            root_dir=args.data_dir,
            batch_size=8,
            transform=transform
        )

        eval_test_set(model, vocab, test_loader)

    else:
        print("Please specify --image for single image or --test for test set evaluation")


if __name__ == '__main__':
    main()
