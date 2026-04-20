"""
==============================================================================
Visual Profiling of KITTI Odometry Sequences Using Multi-Metric Difficulty
Visualization — Graduate Research Project
==============================================================================
Author: [Your Name]
Course: Data Visualization / Robotics
Year:   2025

INSTRUCTIONS:
  1. Set KITTI_BASE_PATH to the root of your KITTI odometry dataset.
     Expected structure:
       <KITTI_BASE_PATH>/
         poses/         00.txt, 01.txt, ... 10.txt
         sequences/
           00/  01/  ...  10/
             times.txt
  2. Run:  python kitti_analysis.py
  3. All figures are saved to C:/Users/syedh/OneDrive/Desktop/EECE-5642-projrect/Outputs/ and displayed interactively.
==============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — EDIT THIS PATH
# ─────────────────────────────────────────────────────────────────────────────
KITTI_BASE_PATH = "C:/Users/syedh/OneDrive/Desktop/Autonomous_viz_project/data/data_odometry_poses/dataset"   # <── point this to your dataset root
FIGURES_DIR     = "C:/Users/syedh/OneDrive/Desktop/EECE-5642-projrect/Outputs"
RANDOM_STATE    = 42
N_CLUSTERS      = 3                    # easy / medium / hard

# Colour palette
CLUSTER_COLORS  = ["#2ECC71", "#F39C12", "#E74C3C"]  # green/orange/red
CLUSTER_LABELS  = ["Easy", "Medium", "Hard"]
ACCENT          = "#2980B9"
DARK            = "#1C2833"
LIGHT           = "#F4F6F7"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_poses(seq_id: str) -> np.ndarray:
    """Load ground-truth poses for a sequence (shape: [N, 3, 4])."""
    pose_file = os.path.join(KITTI_BASE_PATH, "poses", f"{seq_id:02d}.txt")
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.split()))
            mat  = np.array(vals).reshape(3, 4)
            poses.append(mat)
    return np.array(poses)  # [N, 3, 4]


def load_timestamps(seq_id: str) -> np.ndarray | None:
    """Load timestamps (seconds) for a sequence."""
    ts_file = os.path.join(KITTI_BASE_PATH, "sequences", f"{seq_id:02d}", "times.txt")
    if not os.path.exists(ts_file):
        return None
    return np.loadtxt(ts_file)


def get_trajectory(poses: np.ndarray) -> np.ndarray:
    """Extract XZ translation from pose matrices → [N, 2]."""
    return poses[:, [0, 2], 3]   # [X, Z] columns of translation vector


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — METRIC EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_step_distances(traj: np.ndarray) -> np.ndarray:
    """Frame-to-frame Euclidean distances (XYZ) from trajectory."""
    diff = np.diff(traj, axis=0)
    return np.linalg.norm(diff, axis=1)


def compute_rmse(traj: np.ndarray, traj_ref: np.ndarray | None = None) -> float:
    """
    If a second trajectory is supplied (estimated vs. ground-truth) compute
    alignment-free RMSE.  Otherwise compute RMSE of per-step displacements
    vs. a zero-drift baseline (proxy for sequence complexity).
    """
    if traj_ref is not None:
        n = min(len(traj), len(traj_ref))
        errors = np.linalg.norm(traj[:n] - traj_ref[:n], axis=1)
        return float(np.sqrt(np.mean(errors ** 2)))
    dists  = compute_step_distances(traj)
    return float(np.std(dists))  # variance proxy


def compute_drift(traj: np.ndarray) -> tuple[float, float]:
    """
    Linear drift slope and accumulated drift.
    Drift = Euclidean distance from origin as function of frame index.
    Returns (slope, total_accumulated_drift).
    """
    from_origin = np.linalg.norm(traj - traj[0], axis=1)
    x = np.arange(len(from_origin), dtype=float)
    slope = float(np.polyfit(x, from_origin, 1)[0])
    return slope, float(from_origin[-1])


def compute_temporal_metrics(traj: np.ndarray) -> tuple[float, float]:
    """
    Error growth: compare mean displacement in first-half vs. second-half.
    Returns (early_error, late_error).
    """
    n         = len(traj)
    dists     = np.linalg.norm(traj - traj[0], axis=1)
    early_err = float(np.mean(dists[: n // 2]))
    late_err  = float(np.mean(dists[n // 2 :]))
    return early_err, late_err


def compute_curvature(traj: np.ndarray) -> float:
    """
    Mean absolute curvature of the path (proxy for turning complexity).
    Curvature κ = |v × a| / |v|^3  (simplified 2-D cross-product version).
    """
    if traj.shape[1] == 2:
        dx  = np.gradient(traj[:, 0])
        dy  = np.gradient(traj[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        num = np.abs(dx * ddy - dy * ddx)
        den = (dx ** 2 + dy ** 2) ** 1.5 + 1e-10
        return float(np.mean(num / den))
    return 0.0


def compute_motion_entropy(traj: np.ndarray, bins: int = 18) -> float:
    """
    Shannon entropy of heading-angle histogram → heading diversity.
    High entropy ≈ more varied motion directions.
    """
    diff    = np.diff(traj, axis=0)
    angles  = np.arctan2(diff[:, 1], diff[:, 0])
    hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
    prob    = hist / (hist.sum() + 1e-10)
    prob    = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log2(prob)))
    return entropy


def extract_all_metrics(sequence_ids: list[int]) -> pd.DataFrame:
    """Load poses for all sequences and compute all metrics."""
    records = []
    for sid in sequence_ids:
        print(f"  Processing sequence {sid:02d} ...", end=" ")
        t0 = time.time()
        try:
            poses = load_poses(sid)
        except FileNotFoundError as e:
            print(f"SKIPPED ({e})")
            continue

        traj = poses[:, :, 3]          # [N, 3] — full XYZ translation
        traj_xz = traj[:, [0, 2]]     # [N, 2] — XZ plane for 2-D metrics

        rmse               = compute_rmse(traj_xz)
        drift_slope, accum = compute_drift(traj)
        early_err, late_err= compute_temporal_metrics(traj)
        curvature          = compute_curvature(traj_xz)
        entropy            = compute_motion_entropy(traj_xz)
        n_frames           = len(poses)
        runtime            = time.time() - t0  # approximate processing time

        records.append({
            "seq_id"      : sid,
            "label"       : f"Seq {sid:02d}",
            "n_frames"    : n_frames,
            "rmse"        : rmse,
            "drift_slope" : drift_slope,
            "accum_drift" : accum,
            "early_error" : early_err,
            "late_error"  : late_err,
            "curvature"   : curvature,
            "entropy"     : entropy,
            "runtime_s"   : runtime,
        })
        print(f"done  ({n_frames} frames, {runtime:.2f}s)")

    df = pd.DataFrame(records).set_index("seq_id")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — NORMALISATION & CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = ["rmse", "drift_slope", "accum_drift",
               "early_error", "late_error", "curvature", "entropy"]

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df_n   = df.copy()
    df_n[METRIC_COLS] = scaler.fit_transform(df[METRIC_COLS])
    return df_n


def cluster_sequences(df_norm: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    km   = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    lbls = km.fit_predict(df_norm[METRIC_COLS])
    # Reorder clusters by ascending mean RMSE so cluster 0 = easiest
    cluster_rmse = [df_norm.loc[lbls == c, "rmse"].mean() for c in range(n_clusters)]
    order        = np.argsort(cluster_rmse)
    mapping      = {old: new for new, old in enumerate(order)}
    return np.array([mapping[l] for l in lbls])


def compute_difficulty_score(df_norm: pd.DataFrame) -> pd.Series:
    """Composite difficulty score = equal-weighted mean of all normalised metrics."""
    return df_norm[METRIC_COLS].mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Fig 1: Radar (Spider) Chart ───────────────────────────────────────────────

def plot_radar(df_norm: pd.DataFrame, cluster_labels: np.ndarray):
    metrics      = METRIC_COLS
    n_metrics    = len(metrics)
    angles       = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles      += angles[:1]
    labels_nice  = ["RMSE", "Drift Slope", "Accum. Drift",
                    "Early Error", "Late Error", "Curvature", "Entropy"]

    fig, axes = plt.subplots(1, N_CLUSTERS, figsize=(16, 5),
                             subplot_kw=dict(polar=True))
    fig.suptitle("Radar Profile by Difficulty Cluster", fontsize=15, fontweight="bold",
                 y=1.02, color=DARK)

    for c in range(N_CLUSTERS):
        ax     = axes[c]
        seqs   = df_norm[cluster_labels == c]
        color  = CLUSTER_COLORS[c]

        # Mean profile
        mean_vals = seqs[metrics].mean().tolist()
        mean_vals += mean_vals[:1]
        ax.plot(angles, mean_vals, color=color, linewidth=2.5, zorder=3)
        ax.fill(angles, mean_vals, color=color, alpha=0.25, zorder=2)

        # Individual sequences
        for _, row in seqs[metrics].iterrows():
            vals = row.tolist() + row.tolist()[:1]
            ax.plot(angles, vals, color=color, linewidth=0.8, alpha=0.5, zorder=1)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels_nice, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{CLUSTER_LABELS[c]}\n({len(seqs)} sequences)",
                     fontsize=11, fontweight="bold", color=color, pad=18)
        ax.tick_params(labelsize=7)
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig1_radar.png")
    plt.show()
    print("  Saved fig1_radar.png")


# ── Fig 2: Correlation Heatmap ────────────────────────────────────────────────

def plot_correlation_heatmap(df_norm: pd.DataFrame):
    corr         = df_norm[METRIC_COLS].corr()
    mask         = np.triu(np.ones_like(corr, dtype=bool))
    labels_nice  = ["RMSE", "Drift Slope", "Accum.\nDrift",
                    "Early\nError", "Late\nError", "Curvature", "Entropy"]

    cmap = LinearSegmentedColormap.from_list(
        "rb", ["#E74C3C", "#F5F5F5", "#2980B9"], N=256)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, center=0,
                xticklabels=labels_nice, yticklabels=labels_nice,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.75, "label": "Pearson r"},
                ax=ax)
    ax.set_title("Metric Correlation Heatmap\n(Lower Triangle — Pearson r)",
                 fontsize=13, fontweight="bold", color=DARK, pad=14)
    ax.tick_params(axis="both", labelsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig2_heatmap.png")
    plt.show()
    print("  Saved fig2_heatmap.png")


# ── Fig 3: 3-D Scatter Plot ───────────────────────────────────────────────────

def plot_3d_scatter(df_norm: pd.DataFrame, cluster_labels: np.ndarray):
    pca    = PCA(n_components=3, random_state=RANDOM_STATE)
    coords = pca.fit_transform(df_norm[METRIC_COLS])
    var    = pca.explained_variance_ratio_ * 100

    fig  = plt.figure(figsize=(10, 8))
    ax   = fig.add_subplot(111, projection="3d")

    for c in range(N_CLUSTERS):
        mask = cluster_labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                   c=CLUSTER_COLORS[c], s=120, label=CLUSTER_LABELS[c],
                   edgecolors="white", linewidth=0.8, alpha=0.9, zorder=3)
        # Label each point
        for idx, (row_idx, _) in enumerate(df_norm[mask].iterrows()):
            ax.text(coords[np.where(mask)[0][idx], 0],
                    coords[np.where(mask)[0][idx], 1],
                    coords[np.where(mask)[0][idx], 2],
                    f"  {row_idx:02d}", fontsize=7.5, color=DARK)

    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=9, labelpad=8)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=9, labelpad=8)
    ax.set_zlabel(f"PC3 ({var[2]:.1f}%)", fontsize=9, labelpad=8)
    ax.set_title("3-D PCA Scatter — Sequence Clusters\n"
                 f"(Total variance explained: {sum(var):.1f}%)",
                 fontsize=12, fontweight="bold", color=DARK, pad=14)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig3_3d_scatter.png")
    plt.show()
    print("  Saved fig3_3d_scatter.png")


# ── Fig 4: Ranked Bar Plot (Composite Difficulty) ─────────────────────────────

def plot_ranked_bars(df_norm: pd.DataFrame, df_raw: pd.DataFrame,
                     cluster_labels: np.ndarray):
    scores = compute_difficulty_score(df_norm)
    order  = scores.sort_values().index

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_colors = [CLUSTER_COLORS[cluster_labels[df_norm.index.get_loc(i)]] for i in order]
    bars = ax.barh([f"Seq {i:02d}" for i in order], scores[order],
                   color=bar_colors, edgecolor="white", linewidth=0.6, height=0.65)

    # Value labels
    for bar, val in zip(bars, scores[order]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=DARK)

    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=CLUSTER_COLORS[i], label=CLUSTER_LABELS[i])
                  for i in range(N_CLUSTERS)]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Composite Difficulty Score (0 = easy, 1 = hard)", fontsize=10)
    ax.set_title("KITTI Sequence Difficulty Ranking\n(Equal-Weighted Composite Score)",
                 fontsize=13, fontweight="bold", color=DARK, pad=12)
    ax.set_xlim(0, min(1.08, scores.max() * 1.15))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig4_ranked_bars.png")
    plt.show()
    print("  Saved fig4_ranked_bars.png")


# ── Fig 5: Runtime vs Difficulty Scatter ─────────────────────────────────────

def plot_runtime_vs_difficulty(df_norm: pd.DataFrame, df_raw: pd.DataFrame,
                                cluster_labels: np.ndarray):
    scores = compute_difficulty_score(df_norm)
    runtime = df_raw["runtime_s"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(N_CLUSTERS):
        mask = cluster_labels == c
        idxs = df_norm.index[mask]
        ax.scatter(scores[idxs], runtime[idxs],
                   c=CLUSTER_COLORS[c], s=130, label=CLUSTER_LABELS[c],
                   edgecolors="white", linewidth=0.8, zorder=3)
        for i in idxs:
            ax.annotate(f"Seq {i:02d}",
                        (scores[i], runtime[i]),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=8, color=DARK)

    # Trend line
    x, y = scores.values, runtime.values
    z    = np.polyfit(x, y, 1)
    p    = np.poly1d(z)
    xs   = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, p(xs), "--", color="gray", linewidth=1.5, alpha=0.7, label="Trend")

    r, pval = pearsonr(x, y)
    ax.text(0.05, 0.92, f"r = {r:.3f}  (p = {pval:.3f})",
            transform=ax.transAxes, fontsize=9, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.set_xlabel("Composite Difficulty Score", fontsize=11)
    ax.set_ylabel("Processing Runtime (s)", fontsize=11)
    ax.set_title("Runtime vs. Sequence Difficulty\n(Efficiency–Accuracy Tradeoff)",
                 fontsize=13, fontweight="bold", color=DARK, pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig5_runtime_difficulty.png")
    plt.show()
    print("  Saved fig5_runtime_difficulty.png")


# ── Fig 6: Temporal Error Growth ─────────────────────────────────────────────

def plot_temporal_error(df_raw: pd.DataFrame, cluster_labels: np.ndarray,
                        sequence_ids: list[int]):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax_idx, sid in enumerate(sequence_ids):
        if sid not in df_raw.index:
            axes[ax_idx].axis("off")
            continue
        ax      = axes[ax_idx]
        poses   = load_poses(sid)
        traj    = poses[:, :, 3]
        from_o  = np.linalg.norm(traj - traj[0], axis=1)
        frames  = np.arange(len(from_o))
        clabel  = cluster_labels[df_raw.index.get_loc(sid)]
        color   = CLUSTER_COLORS[clabel]

        ax.plot(frames, from_o, color=color, linewidth=1.5)
        ax.fill_between(frames, from_o, alpha=0.15, color=color)
        ax.set_title(f"Seq {sid:02d} — {CLUSTER_LABELS[clabel]}",
                     fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel("Frame", fontsize=7.5)
        ax.set_ylabel("Dist. from start (m)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(linestyle="--", alpha=0.35)

    # Hide any unused axes
    for k in range(len(sequence_ids), len(axes)):
        axes[k].axis("off")

    fig.suptitle("Temporal Error Growth Per Sequence",
                 fontsize=14, fontweight="bold", color=DARK, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig6_temporal_error.png")
    plt.show()
    print("  Saved fig6_temporal_error.png")


# ── Fig 7: Trajectories Grid ─────────────────────────────────────────────────

def plot_trajectories(df_raw: pd.DataFrame, cluster_labels: np.ndarray,
                      sequence_ids: list[int]):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()

    for ax_idx, sid in enumerate(sequence_ids):
        if sid not in df_raw.index:
            axes[ax_idx].axis("off")
            continue
        ax    = axes[ax_idx]
        poses = load_poses(sid)
        traj  = poses[:, [0, 2], 3]   # XZ
        cl    = cluster_labels[df_raw.index.get_loc(sid)]
        color = CLUSTER_COLORS[cl]

        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.2)
        ax.scatter(traj[0, 0], traj[0, 1], color="black", s=40, zorder=5)  # start
        ax.scatter(traj[-1, 0], traj[-1, 1], color="red", s=40, zorder=5)  # end
        ax.set_title(f"Seq {sid:02d}  [{CLUSTER_LABELS[cl]}]",
                     fontsize=9, fontweight="bold", color=color)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Z (m)", fontsize=7)

    for k in range(len(sequence_ids), len(axes)):
        axes[k].axis("off")

    fig.suptitle("Ground-Truth Trajectory Per Sequence  (● start  ● end)",
                 fontsize=13, fontweight="bold", color=DARK, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig7_trajectories.png")
    plt.show()
    print("  Saved fig7_trajectories.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(df_raw: pd.DataFrame, df_norm: pd.DataFrame,
                         cluster_labels: np.ndarray):
    scores = compute_difficulty_score(df_norm)
    # Include ALL metrics so the interactive dashboard can use every column
    df_out = df_raw[["n_frames", "rmse", "drift_slope", "accum_drift",
                      "early_error", "late_error",
                      "curvature", "entropy", "runtime_s"]].copy()
    df_out["difficulty_score"] = scores.round(4)
    df_out["cluster"]          = [CLUSTER_LABELS[c] for c in cluster_labels]

    print("\n" + "=" * 90)
    print("  KITTI SEQUENCE METRICS SUMMARY")
    print("=" * 90)
    print(df_out.to_string())
    print("=" * 90)

    # Export CSV — reset_index() ensures seq_id is a named column (not just the index)
    # so the dashboard can read it as "seq_id" directly.
    df_out.reset_index().rename(columns={"seq_id": "seq_id"}).to_csv(
        f"{FIGURES_DIR}/kitti_metrics_summary.csv", index=False
    )
    print(f"\n  CSV saved → {FIGURES_DIR}/kitti_metrics_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SEQ_IDS = list(range(11))  # 00–10

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   KITTI Odometry — Multi-Metric Difficulty Visualisation  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── 1. Load & compute metrics ──────────────────────────────────────────
    print("[1/5] Extracting metrics from KITTI sequences …")
    df_raw  = extract_all_metrics(SEQ_IDS)
    if df_raw.empty:
        sys.exit("No sequences loaded. Check KITTI_BASE_PATH.")

    # ── 2. Normalise ──────────────────────────────────────────────────────
    print("\n[2/5] Normalising metrics (Min-Max) …")
    df_norm = normalise(df_raw)

    # ── 3. Cluster ────────────────────────────────────────────────────────
    print("[3/5] Clustering sequences (K-Means, k=3) …")
    cluster_labels = cluster_sequences(df_norm)
    for c in range(N_CLUSTERS):
        seqs = df_raw.index[cluster_labels == c].tolist()
        print(f"   {CLUSTER_LABELS[c]:6s}: Seq {seqs}")

    # ── 4. Summary ────────────────────────────────────────────────────────
    print("\n[4/5] Printing summary …")
    print_summary_table(df_raw, df_norm, cluster_labels)

    # ── 5. Visualisations ─────────────────────────────────────────────────
    seq_ids_loaded = df_raw.index.tolist()
    print(f"\n[5/5] Generating {7} figures → {FIGURES_DIR}/ …\n")

    plot_radar(df_norm, cluster_labels)
    plot_correlation_heatmap(df_norm)
    plot_3d_scatter(df_norm, cluster_labels)
    plot_ranked_bars(df_norm, df_raw, cluster_labels)
    plot_runtime_vs_difficulty(df_norm, df_raw, cluster_labels)
    plot_temporal_error(df_raw, cluster_labels, seq_ids_loaded)
    plot_trajectories(df_raw, cluster_labels, seq_ids_loaded)

    print("\n✅  All done!  Check the figures/ directory.")


if __name__ == "__main__":
    main()
