import os
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms
import subprocess
import imageio_ffmpeg

from models.generator import RRDBNet
from infer import infer_tiled

def load_model(checkpoint_path, device, fp16=True):
    print(f"Loading model from {checkpoint_path}...")
    dtype = torch.float16 if fp16 else torch.float32
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
    model.eval().to(device, dtype)
    return model, dtype

def process_video(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dtype = load_model(args.checkpoint, device, args.fp16)
    to_tensor = transforms.ToTensor()
    
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_width = width * 4
    out_height = height * 4
    
    os.makedirs(args.output_dir, exist_ok=True)
    temp_out = os.path.join(args.output_dir, "temp_video.mp4")
    
    # We use mp4v to save the upscaled frames temporarily
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out, fourcc, fps, (out_width, out_height))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames.")
    print("Starting Video Enhancement Pipeline...")
    
    frame_idx = 0
    t0 = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lr = to_tensor(img).unsqueeze(0)
        
        if args.tile > 0 and (lr.shape[2] > args.tile or lr.shape[3] > args.tile):
            sr = infer_tiled(model, lr, 4, args.tile, args.tile_pad, device, dtype)
        else:
            with torch.no_grad():
                sr = model(lr.to(device, dtype)).cpu().float()
                
        sr_img = sr.squeeze(0).clamp(0, 1).numpy()
        sr_img = (sr_img * 255.0).round().astype(np.uint8)
        sr_img = np.transpose(sr_img, (1, 2, 0))
        
        sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
        out.write(sr_img_bgr)
        
        if frame_idx % 10 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            fps_proc = frame_idx / elapsed
            print(f"Processed frame {frame_idx}/{total_frames} ({fps_proc:.2f} fps)")

    cap.release()
    out.release()
    
    print("Upscaling complete. Merging audio using ffmpeg...")
    final_out = os.path.join(args.output_dir, os.path.basename(args.input).rsplit('.', 1)[0] + "_4K.mp4")
    
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        # Use libx264 for the final video to ensure it plays in browsers
        cmd = [
            ffmpeg_exe, "-y",
            "-i", temp_out,
            "-i", args.input,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            final_out
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        os.remove(temp_out)
        print("Audio merge and h264 encoding successful.")
    except Exception as e:
        print(f"Audio merge failed (using raw video): {e}")
        import shutil
        shutil.move(temp_out, final_out)
        
    print(f"Final output saved to: {final_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs/videos")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--tile_pad", type=int, default=32)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    process_video(args)
