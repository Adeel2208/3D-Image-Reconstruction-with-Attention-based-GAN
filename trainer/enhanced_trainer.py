import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from ..losses.perceptual_loss import PerceptualLoss
from ..losses.edge_loss import EdgeAwareLoss
from ..losses.ssim_loss import MultiScaleSSIMLoss
from ..losses.frequency_loss import FrequencyLoss
from ..utils.checkpoint import save_checkpoint

class EnhancedGANTrainer:
    def __init__(self, generator, discriminator, dataloader, device, 
                 lambda_rec=100, lambda_perc=10, lambda_adv=1, lambda_gp=10,
                 lambda_edge=20, lambda_ms_ssim=25, lambda_freq=10,
                 checkpoint_dir='./checkpoints'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.dataloader = dataloader
        self.device = device
        self.lambda_rec = lambda_rec
        self.lambda_perc = lambda_perc
        self.lambda_adv = lambda_adv
        self.lambda_gp = lambda_gp
        self.lambda_edge = lambda_edge
        self.lambda_ms_ssim = lambda_ms_ssim
        self.lambda_freq = lambda_freq
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.criterion_adv = nn.BCEWithLogitsLoss()
        self.criterion_rec = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(device)
        self.edge_loss = EdgeAwareLoss().to(device)
        self.ms_ssim_loss = MultiScaleSSIMLoss().to(device)
        self.frequency_loss = FrequencyLoss().to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

        self.optimizer_g = optim.AdamW(self.generator.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=0.001)
        self.optimizer_d = optim.AdamW(self.discriminator.parameters(), lr=0.0004, betas=(0.9, 0.999), weight_decay=0.001)

        self.scheduler_g = ReduceLROnPlateau(self.optimizer_g, mode='min', factor=0.5, patience=10, verbose=True)
        self.scheduler_d = ReduceLROnPlateau(self.optimizer_d, mode='min', factor=0.5, patience=10, verbose=True)

        self.g_losses = []
        self.d_losses = []
        self.f1_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.accuracy_scores = []
        self.ssim_scores = []
        self.psnr_scores = []

    def gradient_penalty(self, real, fake):
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device, requires_grad=True)
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)
        interpolated_output = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=interpolated_output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_output, device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def train(self, num_epochs=100, checkpoint_interval=10):
        best_ssim = 0.0
        evaluation_metrics = []

        for epoch in range(num_epochs):
            self.generator.train()
            self.discriminator.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            all_preds = []
            all_targets = []
            epoch_ssim = 0.0
            epoch_psnr = 0.0

            for i, (images, masks) in enumerate(self.dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                batch_size = images.size(0)

                self.optimizer_d.zero_grad()
                real_output = self.discriminator(masks)
                real_labels = torch.ones_like(real_output).to(self.device)
                d_real_loss = self.criterion_adv(real_output, real_labels)

                fake_masks = self.generator(images)
                fake_output = self.discriminator(fake_masks.detach())
                fake_labels = torch.zeros_like(fake_output).to(self.device)
                d_fake_loss = self.criterion_adv(fake_output, fake_labels)

                gp = self.gradient_penalty(masks, fake_masks.detach())
                d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gp
                d_loss.backward()
                self.optimizer_d.step()

                self.optimizer_g.zero_grad()
                fake_masks = self.generator(images)
                fake_output = self.discriminator(fake_masks)
                g_adv_loss = self.criterion_adv(fake_output, real_labels)
                g_rec_loss = self.criterion_rec(fake_masks, masks)
                g_perc_loss = self.perceptual_loss(fake_masks, masks)
                g_edge_loss = self.edge_loss(fake_masks, masks)
                g_ms_ssim_loss = self.ms_ssim_loss(fake_masks, masks)
                g_freq_loss = self.frequency_loss(fake_masks, masks)

                g_loss = (self.lambda_adv * g_adv_loss + 
                         self.lambda_rec * g_rec_loss + 
                         self.lambda_perc * g_perc_loss +
                         self.lambda_edge * g_edge_loss +
                         self.lambda_ms_ssim * g_ms_ssim_loss +
                         self.lambda_freq * g_freq_loss)
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.optimizer_g.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()

                preds = (fake_masks > 0.5).float()
                all_preds.append(preds.detach().cpu())
                all_targets.append(masks.detach().cpu())

                epoch_ssim += self.ssim(fake_masks, masks).item()
                epoch_psnr += self.psnr(fake_masks, masks).item()

                if i % 10 == 0:
                    torch.cuda.empty_cache()

            epoch_g_loss /= len(self.dataloader)
            epoch_d_loss /= len(self.dataloader)
            epoch_ssim /= len(self.dataloader)
            epoch_psnr /= len(self.dataloader)

            all_preds = torch.cat(all_preds).numpy()
            all_targets = torch.cat(all_targets).numpy()
            all_preds_bin = (all_preds > 0.5).astype(int).flatten()
            all_targets_bin = (all_targets > 0.5).astype(int).flatten()

            f1 = f1_score(all_targets_bin, all_preds_bin)
            precision = precision_score(all_targets_bin, all_preds_bin, zero_division=0)
            recall = recall_score(all_targets_bin, all_preds_bin, zero_division=0)
            accuracy = accuracy_score(all_targets_bin, all_preds_bin)

            self.g_losses.append(epoch_g_loss)
            self.d_losses.append(epoch_d_loss)
            self.f1_scores.append(f1)
            self.precision_scores.append(precision)
            self.recall_scores.append(recall)
            self.accuracy_scores.append(accuracy)
            self.ssim_scores.append(epoch_ssim)
            self.psnr_scores.append(epoch_psnr)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"D Loss: {epoch_d_loss:.4f} | G Loss: {epoch_g_loss:.4f} | "
                  f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | "
                  f"Accuracy: {accuracy:.4f} | SSIM: {epoch_ssim:.4f} | PSNR: {epoch_psnr:.2f}")

            self.scheduler_g.step(epoch_g_loss)
            self.scheduler_d.step(epoch_d_loss)

            is_best = epoch_ssim > best_ssim
            if is_best:
                best_ssim = epoch_ssim

            if epoch > 7:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                    'optimizer_d_state_dict': self.optimizer_d.state_dict(),
                    'best_ssim': best_ssim,
                }, is_best, filename=checkpoint_path)

        evaluation_metrics = pd.DataFrame({
            'Epoch': range(1, num_epochs + 1),
            'Generator Loss': self.g_losses,
            'Discriminator Loss': self.d_losses,
            'F1 Score': self.f1_scores,
            'Precision': self.precision_scores,
            'Recall': self.recall_scores,
            'Accuracy': self.accuracy_scores,
            'SSIM': self.ssim_scores,
            'PSNR': self.psnr_scores
        })

        print("\n=== Evaluation Metrics ===")
        print(evaluation_metrics.tail(10))

        evaluation_metrics.to_csv(os.path.join(self.checkpoint_dir, 'evaluation_metrics.csv'), index=False)
        self.plot_training(evaluation_metrics)

    def plot_training(self, metrics_df):
        epochs = metrics_df['Epoch']
        plt.figure(figsize=(20, 15))

        plt.subplot(3, 2, 1)
        plt.plot(epochs, metrics_df['Generator Loss'], label='Generator Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator Loss')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(epochs, metrics_df['Discriminator Loss'], label='Discriminator Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss')
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(epochs, metrics_df['F1 Score'], label='F1 Score', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Over Epochs')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(epochs, metrics_df['Precision'], label='Precision', color='purple')
        plt.plot(epochs, metrics_df['Recall'], label='Recall', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision and Recall Over Epochs')
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(epochs, metrics_df['SSIM'], label='SSIM', color='brown')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Structural Similarity Index Measure Over Epochs')
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(epochs, metrics_df['PSNR'], label='PSNR', color='cyan')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('Peak Signal-to-Noise Ratio Over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_curves.png'))
        plt.show()
