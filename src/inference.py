import torch
from PIL import Image
from torchvision import transforms
from train_cnn import LatexCNN
from dataset import create_dataloaders
import os
import random

def predict_latex(model, image_path, vocab, transform, device, max_length=50):

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    pred_caption = [vocab.token_to_idx['<SOS>']]
    
    with torch.no_grad():
        for x in range(max_length):
            caption_tensor = torch.LongTensor(pred_caption).unsqueeze(0).to(device)
            output = model(image, caption_tensor)
            next_token_idx = output[0, -1].argmax().item()
            
            if next_token_idx == vocab.token_to_idx['<EOS>']:
                break
                
            pred_caption.append(next_token_idx)
    
    # convert indices to tokens
    predicted_tokens = [vocab.idx_to_token[idx] for idx in pred_caption[1:]]  # skip <SOS>
    return ' '.join(predicted_tokens)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, 'data', 'crohme_images', 'TC11_CROHME23', 'INKML', 'test')
    
    #same transforms as training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.9], std=[0.1])
    ])
    
    # load vocab from dataset
    _, _, _, vocab = create_dataloaders(
        data_dir,
        batch_size=1,
        transform=transform
    )
    
    # load the model
    if not os.path.exists('model.pth'):
        print("Error: model.pth not found. Please train the model first using train_cnn.py")
        return
        
    model = LatexCNN(len(vocab.token_to_idx)).to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    print("Model loaded successfully")

    available_files = {}
    for root, _, files in os.walk(data_dir):
        for file in files:
            available_files[file] = os.path.join(root, file)

    if(len(available_files) > 0):
        image_path = random.choice(list(available_files.values()))
        print(f"Testing with image: {image_path}")
        predicted_latex = predict_latex(model, image_path, transform, vocab, device)
        print(f"Predicted LaTeX: {predicted_latex}")
    else:
        print("No test images found in the dataset")

if __name__ == '__main__':
    main() 