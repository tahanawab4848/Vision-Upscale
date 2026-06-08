import os
import sys
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Dependency Auto-Installer
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    import subprocess
    print("Installing opencv-python for biometric analysis...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2

# ---------------------------------------------------------------------------
# Mathematical Ridge Pattern Simulator
# ---------------------------------------------------------------------------
def generate_synthetic_fingerprint(size=256):
    """Generates a mathematical concentric whorl fingerprint-like pattern with biometric minutiae."""
    x = np.linspace(-3.5, 3.5, size)
    y = np.linspace(-3.5, 3.5, size)
    X, Y = np.meshgrid(x, y)
    
    # Compute radius and angle
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # Base frequency of ridges
    freq = 5.5
    # Spiral/wavy perturbation to simulate local ridge patterns
    distortion = 0.45 * np.sin(1.8 * r + theta)
    ridges = np.cos(2 * np.pi * freq * (r + distortion))
    
    # Map to 0-255 grayscale
    img = ((ridges + 1.0) * 127.5).astype(np.uint8)
    
    # Add random breaks (white circles) to create ridge endings
    np.random.seed(42)  # For reproducible minutiae points
    for _ in range(12):
        cx = np.random.randint(50, size - 50)
        cy = np.random.randint(50, size - 50)
        cv2.circle(img, (cx, cy), np.random.randint(3, 5), 255, -1)
        
    # Add random bridges (darker lines) to create ridge bifurcations
    for _ in range(12):
        cx = np.random.randint(50, size - 50)
        cy = np.random.randint(50, size - 50)
        dx = np.random.randint(-8, 8)
        dy = np.random.randint(-8, 8)
        cv2.line(img, (cx, cy), (cx + dx, cy + dy), 40, 3)
    
    # Add a mask to make it circular/elliptic like a real fingertip impression
    mask = (X**2 / 9.0 + Y**2 / 12.0) < 1.0
    img[~mask] = 255 # White background
    
    return img

# ---------------------------------------------------------------------------
# Biometric Skeletonization (Ridge Thinning)
# ---------------------------------------------------------------------------
def skeletonize_ridges(binary_img):
    """Skeletonizes a binary image to 1-pixel width ridges using OpenCV morphology."""
    # Smooth to prevent boundary roughness noise
    smoothed = cv2.GaussianBlur(binary_img, (3, 3), 0)
    _, img = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY_INV)
    
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = img.copy()
    
    # Iterative erosion skeletonization
    while True:
        eroded = cv2.erode(temp, element)
        temp2 = cv2.dilate(eroded, element)
        temp2 = cv2.subtract(temp, temp2)
        skel = cv2.bitwise_or(skel, temp2)
        temp = eroded.copy()
        
        if cv2.countNonZero(temp) == 0:
            break
            
    return skel

def prune_minutiae(endings, bifurcations, min_dist=6):
    """Prunes spurious minutiae points that are too close together due to skeleton noise."""
    pruned_endings = []
    pruned_bifs = []
    
    # Filter endings too close to each other
    for e in endings:
        too_close = False
        for pe in pruned_endings:
            if np.sqrt((e[0] - pe[0])**2 + (e[1] - pe[1])**2) < min_dist:
                too_close = True
                break
        if not too_close:
            pruned_endings.append(e)
            
    # Filter bifurcations too close to endings or other bifurcations
    for b in bifurcations:
        too_close = False
        for pb in pruned_bifs:
            if np.sqrt((b[0] - pb[0])**2 + (b[1] - pb[1])**2) < min_dist:
                too_close = True
                break
        if not too_close:
            for pe in pruned_endings:
                if np.sqrt((b[0] - pe[0])**2 + (b[1] - pe[1])**2) < min_dist:
                    too_close = True
                    break
        if not too_close:
            pruned_bifs.append(b)
            
    return pruned_endings, pruned_bifs

# ---------------------------------------------------------------------------
# Biometric Minutiae Points Extraction
# ---------------------------------------------------------------------------
def extract_minutiae(skel_img):
    """Extracts Ridge Endings and Bifurcations using the Crossing Number (CN) algorithm."""
    # Binarize to 0 and 1
    binary = (skel_img > 0).astype(np.uint8)
    
    # Pad to prevent out-of-bounds checks
    pad = np.pad(binary, 1, mode='constant', constant_values=0)
    
    endings = []
    bifurcations = []
    
    h, w = skel_img.shape
    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if pad[y, x] == 1:
                # Retrieve 8-neighborhood clockwise starting from top-left, casting to int to prevent overflow
                n = [
                    int(pad[y-1, x-1]), int(pad[y-1, x]), int(pad[y-1, x+1]),
                    int(pad[y, x+1]), int(pad[y+1, x+1]), int(pad[y+1, x]),
                    int(pad[y+1, x-1]), int(pad[y, x-1])
                ]
                
                # Crossing Number calculation
                cn = 0.5 * sum(abs(n[i] - n[(i+1)%8]) for i in range(8))
                
                # CN = 1 means Ridge Ending (only 1 connection)
                if cn == 1:
                    # Filter out border points to avoid false edge endings
                    if 10 < x < w - 10 and 10 < y < h - 10:
                        endings.append((x - 1, y - 1))
                # CN = 3 means Ridge Bifurcation (3 connections branching out)
                elif cn == 3:
                    if 10 < x < w - 10 and 10 < y < h - 10:
                        bifurcations.append((x - 1, y - 1))
                        
    # Apply biometric minutiae pruning
    endings, bifurcations = prune_minutiae(endings, bifurcations)
    return endings, bifurcations

# ---------------------------------------------------------------------------
# Minutiae Point-Set Matcher
# ---------------------------------------------------------------------------
def match_fingerprints(pts_template, pts_query, dist_threshold=6.0):
    """Computes the percentage match score between two sets of minutiae points."""
    if not pts_template or not pts_query:
        return 0.0
        
    matched_count = 0
    used_indices = set()
    
    for pt1 in pts_template:
        # Find closest unused point in pts_query
        best_dist = float('inf')
        best_idx = -1
        
        for idx, pt2 in enumerate(pts_query):
            if idx in used_indices:
                continue
            dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
                
        if best_dist <= dist_threshold:
            matched_count += 1
            used_indices.add(best_idx)
            
    # Match score = matched points / total template points
    return (matched_count / len(pts_template)) * 100.0

# ---------------------------------------------------------------------------
# Visualizer Plotting Helper
# ---------------------------------------------------------------------------
def draw_minutiae_on_image(img, endings, bifurcations):
    """Draws color markers on fingerprint image representing minutiae."""
    # Convert grayscale to color RGB
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Red circles for Ridge Endings
    for x, y in endings:
        cv2.circle(color_img, (x, y), 3, (0, 0, 255), 1) # Red (BGR)
        
    # Blue squares/markers for Bifurcations
    for x, y in bifurcations:
        cv2.rectangle(color_img, (x - 2, y - 2), (x + 2, y + 2), (255, 0, 0), 1) # Blue (BGR)
        
    return color_img

# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def run_forensic_pipeline():
    print("=" * 60)
    print("FINGERPRINT RESTORATION & MINUTIAE FORENSICS PIPELINE")
    print("=" * 60)
    
    # 1. Generate clean database template fingerprint
    print("[1/6] Generating synthetic high-resolution template fingerprint...")
    template_img = generate_synthetic_fingerprint(size=256)
    cv2.imwrite("fingerprint_template.png", template_img)
    
    # Extract template minutiae
    template_skel = skeletonize_ridges(template_img)
    template_endings, template_bifs = extract_minutiae(template_skel)
    total_template_points = template_endings + template_bifs
    print(f"      -> Extracted {len(template_endings)} endings, {len(template_bifs)} bifurcations.")
    
    # 2. Simulate degraded crime-scene latent print (4x smaller + blurred + noisy)
    print("[2/6] Simulating degraded crime-scene latent fingerprint...")
    # Scale down by 4x (to 64x64)
    lr_size = 64
    lr_img = cv2.resize(template_img, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
    # Add heavy Gaussian blur to smear the ridges
    lr_img_blurred = cv2.GaussianBlur(lr_img, (3, 3), 0)
    # Add pixel noise
    noise = np.random.normal(0, 15, lr_img_blurred.shape).astype(np.int16)
    lr_noisy = np.clip(lr_img_blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite("fingerprint_latent_lr.png", lr_noisy)
    
    # 3. Baseline: Upscale degraded print using standard Bicubic interpolation
    print("[3/6] Upscaling latent print using standard Bicubic interpolation...")
    bicubic_img = cv2.resize(lr_noisy, (256, 256), interpolation=cv2.INTER_CUBIC)
    bicubic_skel = skeletonize_ridges(bicubic_img)
    bic_endings, bic_bifs = extract_minutiae(bicubic_skel)
    total_bic_points = bic_endings + bic_bifs
    
    # 4. Neural: Run ESRGAN upscaler on the latent print (uses our infer.py)
    print("[4/6] Running ESRGAN neural upscaler on latent fingerprint...")
    # We invoke the inference script on fingerprint_latent_lr.png
    # We use the Anime model checkpoint since it restores sharp line contours perfectly
    checkpoint_path = "checkpoints/RealESRGAN_x4plus_anime_6B.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"      -> Warning: {checkpoint_path} not found. Please download it via the GUI first.")
        print("      -> Skipping neural inference phase; matching metrics will show baseline only.")
        has_esrgan = False
    else:
        import subprocess
        cmd = [
            sys.executable, "infer.py",
            "--input", "fingerprint_latent_lr.png",
            "--checkpoint", checkpoint_path,
            "--output", "outputs/infer",
            "--num_blocks", "6",
            "--tile", "128"
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            esrgan_img_path = "outputs/infer/fingerprint_latent_lr_SR.png"
            if os.path.exists(esrgan_img_path):
                esrgan_img = cv2.imread(esrgan_img_path, cv2.IMREAD_GRAYSCALE)
                has_esrgan = True
            else:
                has_esrgan = False
        except Exception as e:
            print(f"      -> ESRGAN Execution failed: {e}")
            has_esrgan = False
            
    # 5. Extract minutiae from ESRGAN image
    if has_esrgan:
        print("[5/6] Extracting minutiae from ESRGAN restored print...")
        esrgan_skel = skeletonize_ridges(esrgan_img)
        esr_endings, esr_bifs = extract_minutiae(esrgan_skel)
        total_esr_points = esr_endings + esr_bifs
    else:
        esr_endings, esr_bifs = [], []
        total_esr_points = []
        esrgan_img = np.ones((256, 256), dtype=np.uint8) * 255
        
    # 6. Biometric Minutiae Matching Comparisons
    print("[6/6] Computing biometric database match scores...")
    score_bicubic = match_fingerprints(total_template_points, total_bic_points)
    
    print("-" * 50)
    print(f"Baseline (Bicubic) Match Accuracy : {score_bicubic:.2f}%")
    if has_esrgan:
        score_esrgan = match_fingerprints(total_template_points, total_esr_points)
        print(f"Neural (ESRGAN) Match Accuracy    : {score_esrgan:.2f}%")
        match_status = "SUCCESSFUL MATCH (PASS)" if score_esrgan >= 70.0 else "INSUFFICIENT DETAILS"
        print(f"Biometric Verification Status    : {match_status}")
    else:
        score_esrgan = 0.0
        print("Neural (ESRGAN) Match Accuracy    : N/A (Checkpoint missing)")
    print("-" * 50)
    
    # 7. Generate beautiful comparison sheet
    print("Generating visual comparison panel: 'fingerprint_comparison.png'...")
    # Create colored representations
    vis_template = draw_minutiae_on_image(template_img, template_endings, template_bifs)
    
    # Resize raw LR to fit size for visualization
    vis_lr = cv2.resize(lr_noisy, (256, 256), interpolation=cv2.INTER_NEAREST)
    vis_lr_color = cv2.cvtColor(vis_lr, cv2.COLOR_GRAY2BGR)
    
    vis_bic = draw_minutiae_on_image(bicubic_img, bic_endings, bic_bifs)
    vis_esr = draw_minutiae_on_image(esrgan_img, esr_endings, esr_bifs) if has_esrgan else np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Draw labels on panel headers
    def draw_label(img, text, pos=(10, 25), color=(0, 255, 0)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    draw_label(vis_template, "Database Template", color=(0, 255, 0))
    draw_label(vis_lr_color, "Latent Print (64x64)", color=(0, 165, 255))
    draw_label(vis_bic, f"Bicubic Match: {score_bicubic:.1f}%", color=(0, 0, 255))
    if has_esrgan:
        draw_label(vis_esr, f"ESRGAN Match: {score_esrgan:.1f}%", color=(0, 255, 0))
    else:
        draw_label(vis_esr, "ESRGAN: Missing Checkpoint", color=(0, 0, 255))
        
    # Stack panels horizontally
    panel = np.hstack((vis_template, vis_lr_color, vis_bic, vis_esr))
    cv2.imwrite("fingerprint_comparison.png", panel)
    print("Done! Open 'fingerprint_comparison.png' to see the side-by-side matching results.")
    print("=" * 60)

if __name__ == "__main__":
    run_forensic_pipeline()
