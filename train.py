import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import pandas as pd
from src.dataset.reconstruction_dataset import ReconstructionDataset
from src.models.generator import HybridVisionTransformerUNet
from src.models.discriminator import PatchDiscriminator
from src.trainer.enhanced_trainer import EnhancedGANTrainer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config('configs/config.yaml')
    
    # Dataset
    transform = transforms.Compose([
        transforms.Resize((config['data']['resize'], config['data']['resize'])),
        transforms.ToTensor(),
    ])
    
    dataset = ReconstructionDataset(
        image_folder=config['data']['image_folder'],
        mask_folder=config['data']['mask_folder'],
        transform=transform,
        augment=config['data']['augment']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Models
    generator = HybridVisionTransformerUNet(
        in_channels=config['model']['generator']['in_channels'],
        out_channels=config['model']['generator']['out_channels'],
        features=config['model']['generator']['features'],
        window_sizes=config['model']['generator']['window_sizes'],
        num_heads=config['model']['generator']['num_heads'],
        depths=config['model']['generator']['depths']
    )
    
    discriminator = PatchDiscriminator(
        in_channels=config['model']['discriminator']['in_channels'],
        features=config['model']['discriminator']['features']
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    
    # Trainer
    trainer = EnhancedGANTrainer(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        lambda_rec=config['loss_weights']['lambda_rec'],
        lambda_perc=config['loss_weights']['lambda_perc'],
        lambda_adv=config['loss_weights']['lambda_adv'],
        lambda_gp=config['loss_weights']['lambda_gp'],
        lambda_edge=config['loss_weights']['lambda_edge'],
        lambda_ms_ssim=config['loss_weights']['lambda_ms_ssim'],
        lambda_freq=config['loss_weights']['lambda_freq'],
        checkpoint_dir=config['training']['checkpoint_dir']
    )
    
    # Train
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        checkpoint_interval=config['training']['checkpoint_interval']
    )
    
    # Evaluation
    metrics_path = os.path.join(config['training']['checkpoint_dir'], 'evaluation_metrics.csv')
    if os.path.exists(metrics_path):
        eval_metrics = pd.read_csv(metrics_path)
        print("\n=== Final Evaluation Metrics ===")
        print(eval_metrics.describe())
    else:
        print("Evaluation metrics file not found.")

if __name__ == "__main__":
    main()
