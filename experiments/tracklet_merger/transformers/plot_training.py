import pandas as pd
import matplotlib.pyplot as plt

files = [
    r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\cross_attention\output\cross_attention_xgb_matched\training_log.csv",
    r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\cross_attention\output\cross_attention_simple_synth\training_log.csv"
]
labels = ["Run A", "Run B"]

runs = [pd.read_csv(f) for f in files]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle("Training Report", fontsize=13)

for i, (df, label) in enumerate(zip(runs, labels)):
    ep = df["epoch"]
    c = f"C{i}"
    axes[0,0].plot(ep, df["train_loss"], color=c, label=f"{label} train")
    axes[0,0].plot(ep, df["val_loss"],   color=c, label=f"{label} val", linestyle="--")
    axes[0,1].plot(ep, df["train_auc"],  color=c, label=f"{label} train")
    axes[0,1].plot(ep, df["val_auc"],    color=c, label=f"{label} val", linestyle="--")
    axes[1,0].plot(ep, df["val_auc"] - df["train_auc"],   color=c, label=f"{label} AUC gap")
    axes[1,0].plot(ep, df["train_loss"] - df["val_loss"], color=c, label=f"{label} loss gap", linestyle="--")
    axes[1,1].plot(ep, df["lr"], color=c, label=label, drawstyle="steps-post")

axes[0,0].set(title="Loss",      xlabel="Epoch", ylabel="Loss")
axes[0,1].set(title="AUC",       xlabel="Epoch", ylabel="AUC")
axes[1,0].set(title="Gen. gap",  xlabel="Epoch", ylabel="Gap")
axes[1,1].set(title="LR",        xlabel="Epoch", ylabel="Learning rate")
axes[1,0].axhline(0, color="black", linewidth=0.8, linestyle=":")

for ax in axes.flat:
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.savefig("training_report.png", dpi=150)
plt.show()