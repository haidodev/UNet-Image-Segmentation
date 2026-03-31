import torch
import torch.nn as nn
from training.dataset import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from training.utils import get_model, dice_loss, dice_score
import datetime
import argparse

parser = argparse.ArgumentParser(description="Train UNet variants on Oxford-IIIT Pet Dataset")
parser.add_argument("--model", type=str, required=True, choices=["unet", "unetpp", "attention_unet", "deeplab", "transunet"], help="Model architecture to train")
parser.add_argument("--img_size", type=int, default=128, help="Image size for training")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()
writer = SummaryWriter(log_dir=f"runs/{args.model}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

def train(model, train_loader, val_loader, device, criterion, optimizer, epochs, output_path):
    best_val_dice = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device).float()   
            outputs = model(images)
            bce = criterion(outputs, masks)
            d_loss = dice_loss(outputs, masks)
            loss = bce + d_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_log = train_loss / len(train_loader)
        
        
        model.eval()
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).float()
                outputs = model(images)
                val_dice += dice_score(outputs, masks).item()
        
        val_dice /= len(val_loader)
        writer.add_scalar("Loss/train", avg_log, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)
        
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), output_path)
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_log:.4f}, Val Dice: {val_dice:.4f}")
        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloader(
        img_size=args.img_size, 
        batch_size=args.batch_size, 
        val_split=args.val_split, 
        seed=args.seed
    )
    
    model = get_model(args.model).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint_path = f"best_{args.model}_{args.img_size}.pth"
    train(model, train_loader, val_loader, device, criterion, optimizer, epochs=args.epochs, output_path=checkpoint_path)
    
if __name__ == "__main__":
    main()