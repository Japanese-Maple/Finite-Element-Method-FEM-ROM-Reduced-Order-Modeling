import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import itertools
from sklearn.decomposition import PCA

# --- Helper Functions ---
def get_hypercube_edges(corners):
    """Calculate all edges of an N-dimensional hypercube."""
    connections = []
    for i, j in itertools.combinations(range(len(corners)), 2):
        if np.sum(corners[i] != corners[j]) == 1:
            connections.append((i, j))
    return connections

def rotation_matrix_5d(theta, axis1, axis2):
    """Generate a 5D rotation matrix for a given plane and angle."""
    R = np.eye(5)
    R[axis1, axis1] = np.cos(theta)
    R[axis2, axis2] = np.cos(theta)
    R[axis1, axis2] = -np.sin(theta)
    R[axis2, axis1] = np.sin(theta)
    return R

# --- Generate Base Hypercube ---
bounds = [10, 200]
corners_5d = np.array(list(itertools.product(bounds, repeat=5)), dtype=float)
center_5d = np.mean(corners_5d, axis=0)

# Calculate connections (80 edges for a 5D hypercube)
edge_connections = get_hypercube_edges(corners_5d)

# --- Initial Projection Setup ---
# We fit PCA once to get a fixed viewing plane, preventing the view 
# from "snapping" around when we rotate the shape.
pca = PCA(n_components=2)
pca.fit(corners_5d)
proj_matrix = pca.components_  # Fixed 2x5 projection matrix
pca_mean = pca.mean_

# --- Interactive Plot Setup ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.35) # Make room for sliders

# Initial scatter and line objects (we will update their data in the callback)
init_2d = (corners_5d - pca_mean) @ proj_matrix.T
scatter = ax.scatter(init_2d[:, 0], init_2d[:, 1], c='blue', s=55, alpha=0.8, zorder=3)

lines = []
for i, j in edge_connections:
    line, = ax.plot([init_2d[i, 0], init_2d[j, 0]], 
                    [init_2d[i, 1], init_2d[j, 1]], 
                    'k-', lw=1.5, zorder=1, alpha=0.6)
    lines.append((line, i, j))

# Formatting
ax.set_title('Interactive 5D Hypercube Rotation', fontsize=17)
ax.set_xlabel('PC1', fontsize=14)
ax.set_ylabel('PC2', fontsize=14)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7, zorder=0)

# Fix limits so the box doesn't resize during rotation
max_radius = np.linalg.norm(corners_5d[0] - center_5d)
ax.set_xlim(-max_radius, max_radius)
ax.set_ylim(-max_radius, max_radius)

# --- Sliders Setup ---
axcolor = 'lightgoldenrodyellow'
slider_axes = [plt.axes([0.15, 0.25 - i*0.04, 0.7, 0.03], facecolor=axcolor) for i in range(5)]
sliders = []
planes = [(0,1), (1,2), (2,3), (3,4), (4,0)]
labels = ['Rot XY (0-1)', 'Rot YZ (1-2)', 'Rot ZW (2-3)', 'Rot WV (3-4)', 'Rot VX (4-0)']

for ax_s, label in zip(slider_axes, labels):
    slider = Slider(ax_s, label, 0.0, 2*np.pi, valinit=0.0, valstep=0.05)
    sliders.append(slider)

# --- Update Function ---
def update(val):
    # Retrieve angles from sliders
    angles = [s.val for s in sliders]
    
    # Compose the 5D rotation matrix
    R_total = np.eye(5)
    for angle, (ax1, ax2) in zip(angles, planes):
        R_total = R_total @ rotation_matrix_5d(angle, ax1, ax2)
        
    # Rotate the corners around the center
    centered_corners = corners_5d - center_5d
    rotated_corners = centered_corners @ R_total.T + center_5d
    
    # Project down to 2D using the fixed PCA plane
    corners_2d = (rotated_corners - pca_mean) @ proj_matrix.T
    
    # Update plot data
    scatter.set_offsets(corners_2d)
    for line, i, j in lines:
        line.set_data([corners_2d[i, 0], corners_2d[j, 0]], 
                      [corners_2d[i, 1], corners_2d[j, 1]])
        
    fig.canvas.draw_idle()

# Attach the update function to all sliders
for slider in sliders:
    slider.on_changed(update)

plt.show()