from collections import defaultdict
import matplotlib.pyplot as plt

KIND_COLORS = {"EPR": "#2a78d6", "swap": "#eb6834", "two-qubit": "#1baf7a",
               "classical": "#eda100", "single-qubit": "#b0afa9"}

def kind_of(name):
    if name == "EPR": return "EPR"
    if name == "swap": return "swap"
    if name in ("cp", "cx", "cz"): return "two-qubit"
    if name in ("measure", "reset", "x", "z"): return "classical"  # corrections
    return "single-qubit"

def plot_schedule(qc, schedule, makespan, save_path=None):
    lane = {q: i for i, q in enumerate(qc.qubits)}
    fig, ax = plt.subplots(figsize=(14, 0.32 * len(qc.qubits)))
    for op in schedule:
        c = KIND_COLORS[kind_of(op["name"])]
        for q in op["qubits"]:
            ax.broken_barh([(op["start"], op["finish"] - op["start"])],
                           (lane[q] + 0.15, 0.7), facecolors=c, edgecolor="none")
    ax.set_yticks(range(len(qc.qubits)))
    ax.set_yticklabels([f"{qc.find_bit(q).registers[0][0].name}[{qc.find_bit(q).registers[0][1]}]"
                        for q in qc.qubits], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("time (arbitrary units)")
    ax.set_xlim(0, makespan)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in KIND_COLORS.values()]
    ax.legend(handles, KIND_COLORS.keys(), loc="upper right", fontsize=8)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300)
