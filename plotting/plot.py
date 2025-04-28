from matplotlib import pyplot as plt
from typing import List, Tuple, Dict

def extract(key: str, item: Dict[float]) -> float:
    value = item[key]
    return value.cpu().item() if hasattr(value, 'cpu') else value


def plot_results_panels(losses: List[Dict], plot_configs: List[Tuple], figsize: Tuple[int, int] = (16, 4)) -> plt.Figure:
    fig, axs = plt.subplots(1, len(plot_configs), figsize=figsize, sharex=True)

    epochs = [item['Epoch'] for item in losses]

    for ax, (key, title, color) in zip(axs, plot_configs):
        values = [extract(key, item) for item in losses]
        ax.plot(epochs, values, label=key, color=color)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    
    return fig