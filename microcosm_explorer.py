import os
import time
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

from models.generator import RRDBNet

def parse_args():
    parser = argparse.ArgumentParser(description="Microcosm Explorer: Infinite ESRGAN Zoom")
    parser.add_argument("--input", type=str, required=True, help="Path to the starting image")
    parser.add_argument("--output_dir", type=str, default="outputs/microcosm", help="Directory to save frames")
    parser.add_argument("--levels", type=int, default=3, help="Number of recursive 4x zoom levels")
    parser.add_argument("--frames_per_level", type=int, default=60, help="Frames per zoom level")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/RealESRGAN_x4plus.pth", help="Model checkpoint")
    parser.add_argument("--resolution", type=int, default=512, help="Output resolution (square)")
    return parser.parse_args()

def center_crop(img, crop_w, crop_h):
    w, h = img.size
    left = (w - crop_w) / 2
    top = (h - crop_h) / 2
    right = (w + crop_w) / 2
    bottom = (h + crop_h) / 2
    return img.crop((left, top, right, bottom))

def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    model = RRDBNet(in_channels=3, out_channels=3, num_features=64, num_blocks=23, scale=4)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("params_ema", ckpt.get("generator", ckpt))
    
    new_state = {}
    for k, v in state.items():
        k = k.replace("RRDB_trunk.", "body.")
        k = k.replace(".RDB1.", ".db1.")
        k = k.replace(".RDB2.", ".db2.")
        k = k.replace(".RDB3.", ".db3.")
        k = k.replace(".rdb1.", ".db1.")
        k = k.replace(".rdb2.", ".db2.")
        k = k.replace(".rdb3.", ".db3.")
        k = k.replace("trunk_conv.", "conv_body.")
        k = k.replace("upconv1.", "upsample.1.")
        k = k.replace("upconv2.", "upsample.4.")
        k = k.replace("conv_up1.", "upsample.1.")
        k = k.replace("conv_up2.", "upsample.4.")
        k = k.replace("HRconv.", "conv_hr.")
        new_state[k] = v
    
    model.load_state_dict(new_state, strict=True)
    model.eval().to(device)
    return model

def upscale_image(model, img, device):
    to_tensor = transforms.ToTensor()
    lr = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        sr = model(lr)
    sr = sr.squeeze(0).cpu().clamp(0, 1).numpy()
    sr = (sr * 255.0).round().astype(np.uint8)
    sr = np.transpose(sr, (1, 2, 0))
    return Image.fromarray(sr)

def generate_microcosm(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(args.checkpoint, device)
    
    print(f"Loading initial image: {args.input}")
    current_img = Image.open(args.input).convert("RGB")
    
    w, h = current_img.size
    min_dim = min(w, h)
    current_img = center_crop(current_img, min_dim, min_dim)
    current_img = current_img.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
    
    frame_count = 0
    print("\nStarting recursive hallucination journey...")
    
    for level in range(args.levels):
        print(f"\n--- Level {level + 1}/{args.levels} ---")
        print("Upscaling 4x (Inferring new microscopic details)...")
        t0 = time.time()
        upscaled_img = upscale_image(model, current_img, device)
        print(f"Upscaling complete in {time.time() - t0:.2f}s")
        
        print("Generating smooth zoom frames...")
        for f in range(args.frames_per_level):
            t = f / args.frames_per_level
            zoom = 4 ** t
            
            crop_size = args.resolution * 4 / zoom
            
            cropped_frame = center_crop(upscaled_img, crop_size, crop_size)
            final_frame = cropped_frame.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
            
            frame_filename = os.path.join(args.output_dir, f"frame_{frame_count:05d}.png")
            final_frame.save(frame_filename)
            frame_count += 1
            
            if f % 15 == 0:
                print(f"  Saved frame {frame_count}")
        
        current_img = center_crop(upscaled_img, args.resolution, args.resolution)
        
    print("\nJourney complete!")
    print(f"Saved {frame_count} frames to '{args.output_dir}'")
    print("To compile into a video, use ffmpeg:")
    print(f"ffmpeg -framerate 30 -i {args.output_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p microcosm_trip.mp4")

if __name__ == "__main__":
    generate_microcosm(parse_args())
