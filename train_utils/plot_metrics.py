# train_utils/plot_metrics.py
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def plot_train_val_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot losses (solid)
    ax1.plot(epochs, train_losses, label="$Loss_{Train}$", color="black", linestyle='-')
    ax1.plot(epochs, val_losses, label="$Loss_{Val}$", color="black", linestyle='--')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # Plot accuracies (dashed)
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accs, label="$Acc_{Train}$", color="green", linestyle='-')
    ax2.plot(epochs, val_accs, label="$Acc_{Val}$", color="orange", linestyle='--')
    ax2.set_ylabel("Accuracy")

    # Title, grid, legend
    ax1.set_title("Train/Validation Loss and Accuracy Over Epochs")
    ax1.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()

    # Save
    fig_path_png = os.path.join(output_dir, "loss_accuracy_plot.png")
    fig_path_pdf = os.path.join(output_dir, "loss_accuracy_plot.pdf")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    print(f"📊 Plots saved to:\n - {fig_path_png}\n - {fig_path_pdf}")
    plt.show()
    plt.close()




def plot_loss_accuracy(loss_list,
                        loss_energy_list,
                        loss_alpha_list,
                        loss_q0_list,
                        acc_list,
                        acc_energy_list,
                        acc_alpha_list,
                        acc_q0_list,
                        output_dir,
                        title=""):

    epochs = np.arange(1, len(loss_list) + 1)
    colors = {
        "total": "black",
        "energy": "red",
        "alpha": "blue",
        "q0": "green",
    }

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Losses
    ax1.plot(epochs, loss_list, label="$Loss_{Total}$", color=colors["total"], linestyle='-')
    ax1.plot(epochs, loss_energy_list, label="$Loss_{Energy}$", color=colors["energy"], linestyle='-')
    ax1.plot(epochs, loss_alpha_list, label="$Loss_{{\\alpha}_s}$", color=colors["alpha"], linestyle='-')
    ax1.plot(epochs, loss_q0_list, label="$Loss_{Q_0}$", color=colors["q0"], linestyle='-')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Plot Accuracies
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc_list, label="$Acc_{Total}$", color=colors["total"], linestyle='--')
    ax2.plot(epochs, acc_energy_list, label="$Acc_{Energy}$", color=colors["energy"], linestyle='--')
    ax2.plot(epochs, acc_alpha_list, label="$Acc_{{\\alpha}_s}$", color=colors["alpha"], linestyle='--')
    ax2.plot(epochs, acc_q0_list, label="$Acc_{Q_0}$", color=colors["q0"], linestyle='--')
    ax2.set_ylabel("Accuracy")

    # Title and Legend
    ax1.set_title(f"{title}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()

    
    plot_file_name = title.lower().replace(" ", "_").replace(":", "").replace(",", "").replace("(", "").replace(")", "")

    # Save Plots
    png_path = os.path.join(output_dir, f"{plot_file_name}_plot.png")
    pdf_path = os.path.join(output_dir, f"{plot_file_name}_plot.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"📉 Loss plot saved as:\n  - {png_path}\n  - {pdf_path}")
    plt.show()
    plt.close()

import matplotlib as mpl
#  --- Global plot style ---
mpl.rcParams.update({
    "font.size": 15,           # global font size
    "axes.titlesize": 15,      # title font size
    "axes.labelsize": 15,      # x and y labels
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 16,
    "image.cmap": "Oranges",   # set global colormap
    "axes.prop_cycle": plt.cycler(color=["orange"]),  # <- all bars/lines orange
    # --- default edge color for patches (bars, hist bars, etc.) ---
    "patch.edgecolor": "black",

    # layout / suptitle spacing
    "figure.autolayout": False,   # keep manual control
    "figure.subplot.top": 0.88    # controls suptitle y position (0–1)
})
# when calling bar/hist, force alpha separately
default_alpha = 0.7
def plot_confusion_matrices(metrics_dict, output_dir,color_map="Oranges"):
    """
    Plots and saves confusion matrices from a best_metrics-style dict.
    """
    output_dir = os.path.join(output_dir, "confusion_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = {
        "energy": ["MATTER", "MATTER-LBT"],
        "alpha": ["0.2", "0.3", "0.4"],
        "q0": ["1.0", "1.5", "2.0", "2.5"]
    }

    for task, labels in tasks.items():
        cm = np.array(metrics_dict[task]["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(cmap=color_map, ax=ax, values_format="d", colorbar=False)
        set_confusion_matrix_title(task, ax)
        # ax.set_title(f"{task.upper()} Confusion Matrix")
        fig.tight_layout()

        # Save
        png_path = os.path.join(output_dir, f"confusion_matrix_{task}.png")
        pdf_path = os.path.join(output_dir, f"confusion_matrix_{task}.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"✅ Saved confusion matrix for {task}:\n - {png_path}\n - {pdf_path}")
        plt.show()
        plt.close()

def set_confusion_matrix_title(task, ax):
    if task == "energy":
        title = "Energy Loss Module"
    elif task == "alpha":
        title = r"Strong Coupling ($\alpha_s$)"
    elif task == "q0":
        title = r"Virtuality Separation ($Q_0$)"
    else:
        title = f"{task} Confusion Matrix"

    ax.set_title(f"{title} Confusion Matrix")


def plot_colormap_list():
    """
    Plots a list of colormaps from matplotlib in various categories.
    """
    colormap_categories = {
        "Perceptually Uniform Sequential": ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
        "Sequential": ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                       'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                       'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'],
        "Diverging": ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'],
        "Qualitative": ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'],
        "Cyclic": ['twilight', 'twilight_shifted', 'hsv'],
        "Miscellaneous": ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                          'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                          'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
    }

    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    for category, cmap_list in colormap_categories.items():
        n = len(cmap_list)
        fig, axes = plt.subplots(n, 1, figsize=(8, 0.4 * n))
        fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
        fig.suptitle(category, fontsize=12, x=0.5, y=1.05)

        if n == 1:
            axes = [axes]

        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            ax.set_axis_off()
            ax.set_title(name, fontsize=10, loc='left')

        plt.show()

# plot_colormap_list()

