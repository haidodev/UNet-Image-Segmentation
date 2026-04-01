import torch
from dataset import get_test_dataloader
from utils import get_model, dice_score
import argparse

parser = argparse.ArgumentParser(description="Evaluate UNet variants on Oxford-IIIT Pet Dataset")
parser.add_argument("--model", type=str, required=True, choices=["unet", "unetpp", "attention_unet", "deeplab", "transunet"], help="Model architecture to evaluate")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
parser.add_argument("--img_size", type=int, default=128, help="Image size for evaluation")
args = parser.parse_args()

        
def evaluate(model, test_loader, device):
    model.eval()
    test_dice = 0.0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            test_dice += dice_score(outputs, masks).item()
    
    test_dice /= len(test_loader)
    print(f"Test Dice Score: {test_dice:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    test_loader = get_test_dataloader(batch_size=8, img_size=args.img_size)
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()