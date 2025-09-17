import cv2
import numpy as np
from scipy.interpolate import splprep, splev

# -----------------------------
# STEP 1: Analyze Kolam Image
# -----------------------------
def analyze_kolam_image(img_path, min_points=30):
    """
    Detect contours (Kolam lines) and dots from a Kolam image.
    Uses skeletonization to ensure continuous smooth paths.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # --- Threshold for pattern (white lines as foreground) ---
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Skeletonize to thin the lines (1-pixel wide) ---
    skeleton = cv2.ximgproc.thinning(binary)

    # --- Extract contours from skeleton ---
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter out very tiny noise
    contours = [c for c in contours if len(c) > min_points]

    # --- Detect dots using HoughCircles ---
    dots = []
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=3, maxRadius=20
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x,y,r) in circles[0,:]:
            dots.append((x,y))

    return contours, img.shape, dots


# -----------------------------
# STEP 2: Smooth contours
# -----------------------------
def smooth_contour(contour, smooth_factor=2):
    """Use B-spline to smooth a contour"""
    contour = contour[:,0,:]  # shape (N,2)
    if len(contour) < 4:
        return contour.reshape(-1,1,2)
    x, y = contour[:,0], contour[:,1]
    try:
        tck, u = splprep([x, y], s=smooth_factor, per=1)
        u_new = np.linspace(0, 1, len(x)*3)
        x_new, y_new = splev(u_new, tck)
        smooth = np.stack([x_new, y_new], axis=-1).reshape(-1,1,2).astype(np.int32)
        return smooth
    except:
        return contour.reshape(-1,1,2)


# -----------------------------
# STEP 3: Center contours + dots
# -----------------------------
def center_contours(contours, dots, canvas_size):
    """Shift both contours and dots so they are centered"""
    h, w = canvas_size[0], canvas_size[1]
    all_points = np.vstack([c.reshape(-1,2) for c in contours])
    if dots:
        all_points = np.vstack([all_points, np.array(dots)])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    shape_w, shape_h = x_max - x_min, y_max - y_min

    dx = (w - shape_w)//2 - x_min
    dy = (h - shape_h)//2 - y_min

    shifted_contours = [c + np.array([dx, dy]) for c in contours]
    shifted_dots = [(x+dx, y+dy) for (x,y) in dots]
    return shifted_contours, shifted_dots

# -----------------------------
# STEP 4: Pipeline
# -----------------------------
def kolam_pipeline(img_path):
    contours, img_shape, dots = analyze_kolam_image(img_path)
    print(f"Detected {len(contours)} contours and {len(dots)} dots.")

    smooth_contours = [smooth_contour(c) for c in contours]
    centered_contours, centered_dots = center_contours(smooth_contours, dots, img_shape)
    print("Contours and dots centered.")

if __name__ == "__main__":
    kolam_pipeline("kolam.png")
