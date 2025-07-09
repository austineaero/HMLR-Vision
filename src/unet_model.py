import os
import torch
import segmentation_models_pytorch as smp
import numpy as np

def build_unet(encoder="resnet18", classes=3, in_channels=3, pretrained=True):
    """Build a U-Net model with configurable input channels."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=classes
    )

def train_unet(model, dataloader, device, epochs=5, lr=1e-4, save_path=None):
    """Train U-Net with CrossEntropyLoss and return epoch loss history."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device).train()
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 0
        
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device).squeeze(1).long()
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved ➜ {save_path}")
    
    return model, epoch_losses

def infer_masks(model, loader, device, thresh=0.5):
    """Run inference and return boundary and text masks."""
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(loader))
        imgs = imgs.to(device)
        logits = model(imgs)[0].cpu() # [C,H,W]
        probs = torch.softmax(logits, dim=0).numpy()
        # Class predictions
        pred_mask = np.argmax(probs, axis=0)
        # Split masks for output
        boundary_mask = (pred_mask == 1).astype(np.uint8)
        text_mask = (pred_mask == 2).astype(np.uint8)
    return boundary_mask, text_mask

def load_unet(weights_path, encoder="resnet18", classes=3, in_channels=3, device="cpu"):
    """Instantiate U-Net and load weights."""
    model = build_unet(encoder=encoder, classes=classes, in_channels=in_channels)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()