import os
import sys
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import load_predictor
from src.data_loader import load_config

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_batch, target_class=None):
        # Forward pass
        logits = self.model(input_batch)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        logits[0, target_class].backward()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # ReLU on the heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap
        heatmap /= torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy(), target_class

def get_target_layer(model, architecture):
    if architecture == "resnet50":
        return model.backbone.layer4[-1]
    elif architecture == "efficientnet_v2":
        return model.backbone.features[-1]
    elif architecture == "swin_transformer":
        # Swin Transformer is a bit tricky, torchvision's implementation
        # uses features as a Sequential. The last block is usually index -1.
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unsupported architecture for Grad-CAM: {architecture}")

def show_results(results):
    """Display processed images using an interactive viewer."""
    if not results:
        return
        
    # Check if matplotlib can actually show anything
    backend = matplotlib.get_backend().lower()
    if 'agg' not in backend:
        try:
            num_results = len(results)
            cols = min(4, num_results)
            rows = (num_results + cols - 1) // cols
            
            plt.figure(figsize=(16, 4 * rows))
            plt.suptitle("Batch Inference Results", fontsize=16, fontweight='bold', color='#2c3e50')
            
            for i, res in enumerate(results):
                plt.subplot(rows, cols, i + 1)
                img = cv2.imread(res['path'])
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.title(f"{res['label']} ({res['confidence']*100:.1f}%)", fontsize=10)
                else:
                    plt.text(0.5, 0.5, "Load Error", ha='center', va='center')
                plt.axis('off')
                
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            print("[Batch] Opening Matplotlib viewer...")
            plt.show()
            return
        except Exception:
            pass

    # Fallback to OpenCV if matplotlib fails or is non-interactive
    print("[Batch] Matplotlib non-interactive. Falling back to OpenCV viewer...")
    print("[Batch] Press ANY KEY to see next image, 'q' or ESC to quit.")
    
    cv2.namedWindow("DermAssist Results", cv2.WINDOW_NORMAL)
    
    for i, res in enumerate(results):
        img = cv2.imread(res['path'])
        if img is not None:
            # Show image first to create the window
            cv2.imshow("DermAssist Results", img)
            
            # Set title after window is created
            title = f"Result {i+1}/{len(results)}: {res['label']} ({res['confidence']*100:.1f}%) - Press any key"
            cv2.setWindowTitle("DermAssist Results", title)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break
        else:
            print(f"[Warning] Could not load {res['path']} for viewing")
            
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Batch Inference with Localization")
    parser.add_argument("--input_dir", type=str, default="data/test", help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Directory to save results")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--show", action="store_true", help="Display results in a python image viewer")
    args = parser.parse_args()

    # Load config and predictor
    config = load_config(args.config)
    predictor = load_predictor(config_path=args.config)
    model = predictor.model
    architecture = predictor.architecture
    
    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup Grad-CAM
    try:
        target_layer = get_target_layer(model, architecture)
        grad_cam = GradCAM(model, target_layer)
        print(f"[Batch] Grad-CAM initialized for {architecture}")
    except Exception as e:
        print(f"[Warning] Could not initialize Grad-CAM: {e}")
        grad_cam = None

    # Get images
    image_paths = glob.glob(os.path.join(args.input_dir, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not image_paths:
        print(f"[Error] No images found in {args.input_dir}")
        return

    print(f"[Batch] Found {len(image_paths)} images. Starting processing...")

    processed_results = []

    for img_path in tqdm(image_paths):
        try:
            # 1. Run standard prediction
            result = predictor.predict(img_path)
            label = result['label']
            confidence = result['confidence']
            
            # 2. Run Grad-CAM if enabled
            if grad_cam:
                # We need the input tensor again for Grad-CAM (since we need gradients)
                image = Image.open(img_path).convert("RGB")
                input_tensor = predictor.transform(image)
                input_batch = input_tensor.unsqueeze(0).to(predictor.device)
                input_batch.requires_grad = True
                
                heatmap, class_idx = grad_cam.generate_cam(input_batch)
                
                # 3. Process heatmap and draw circle
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    # Fallback for formats cv2 might not like (webp)
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                h, w, _ = img_cv.shape
                
                # Resize heatmap to match image size
                heatmap_resized = cv2.resize(heatmap, (w, h))
                heatmap_resized = np.uint8(255 * heatmap_resized)
                
                # Find centroid of the hottest region
                # Use a threshold to find the "peak" area
                _, thresh = cv2.threshold(heatmap_resized, 200, 255, cv2.THRESH_BINARY)
                M = cv2.moments(thresh)
                
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    # Fallback to max location
                    _, _, _, max_loc = cv2.minMaxLoc(heatmap_resized)
                    cX, cY = max_loc
                
                # 4. Generate Heatmap Overlay (Better visualization)
                # Apply JET colormap to heatmap
                heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                
                # Blend with original image
                # We use the original img_cv (before drawing on it)
                img_display = cv2.addWeighted(img_cv, 0.7, heatmap_color, 0.3, 0)
                
                # Draw red circle on the display image
                radius = int(min(h, w) * 0.08)
                cv2.circle(img_display, (cX, cY), radius, (0, 0, 255), 3)
                
                # Add label text
                text = f"{label} ({confidence*100:.1f}%)"
                cv2.putText(img_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Save result
                base_name = os.path.basename(img_path)
                save_path = os.path.join(args.output_dir, f"result_{base_name}")
                cv2.imwrite(save_path, img_display)
                
                # Store for viewing
                processed_results.append({
                    'path': save_path,
                    'label': label,
                    'confidence': confidence
                })
            else:
                # Just copy or save without circle if Grad-CAM failed
                print(f"[Skip] Localization failed for {img_path}")
                
        except Exception as e:
            print(f"[Error] Failed to process {img_path}: {e}")

    print(f"\n[Batch] Completed! Results saved to: {args.output_dir}")

    # Show results if requested
    if args.show:
        show_results(processed_results)

if __name__ == "__main__":
    main()
