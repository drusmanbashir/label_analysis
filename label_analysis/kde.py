
# %%
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def kde(csv_fn):
        df = pd.read_csv(csv_fn)

        xcol = "major_axis"
        ycol = "volume_cm3"

        df= df[df["processing_error"]!=True]
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[xcol, ycol])
        df = df[(df[xcol] > 0) & (df[ycol] > 0)].copy()

# 4. extract arrays
        x = df[xcol].to_numpy()
        y = df[ycol].to_numpy()
        y_plot = np.log10(y)
        y_plot  =y

        values = np.vstack([x,y_plot])
        kde= gaussian_kde(values)
# 6. make grid
        xg = np.linspace(x.min(), x.max(), 100)
        yg = np.linspace(y_plot.min(), y_plot.max(), 100)
        X, Y = np.meshgrid(xg, yg)

        grid_coords = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(grid_coords).reshape(X.shape)

# # 7. plot surface
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection="3d")
#
#     ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
#
#     ax.set_xlabel("major_axis")
#     ax.set_ylabel(y_label)
#     ax.set_zlabel("density")
#     ax.set_title("Surface KDE of lesion length vs volume")
#
        plt.figure(figsize=(7, 6))
        plt.scatter(x, y_plot, s=8, alpha=0.3)
        cs = plt.contour(X, Y, Z, levels=15, colors="red")
        plt.clabel(cs, inline=True, fontsize=8)
        plt.xlabel("major_axis")
        plt.ylabel(y_label)
        plt.title("KDE contours with lesions")
        plt.show()

        # plt.tight_layout()
        # plt.show()
# %%
if __name__ == '__main__':
    

    csv_fn = Path(
        "/r/datasets/preprocessed/lidc/lbd/spc_075_075_075_rlb109adb5e_rlb109adb5e_ex000/label_stats/lesion_stats.csv"
    )
    csv_fn  = "/media/UB/datasets/kits23/label_analysis/lesion_stats.csv"

    csv_lidc="/media/UB/datasets/lidc_all/label_analysis/lesion_stats.csv"
    kde(csv_fn)
    kde(csv_lidc)


    df = pd.read_csv(csv_lidc)
    df['volume_cm3'] = df['volume_mm3']/1000
    df.to_csv(csv_lidc, index=False)
# %%
