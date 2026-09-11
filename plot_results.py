from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from data_processing import interpolate_profiles
from model import properties_q23


def set_plot_style():
    try:
        font_path = font_manager.findfont("SimSun", fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError("未找到宋体字体，请先安装SimSun后再生成论文图") from error
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", font_name],
            "font.size": 9,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
            "savefig.dpi": 600,
        }
    )


def finish_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.75)
    axis.set_axisbelow(True)


def plot_boundary(boundary, output_root=Path("outputs")):
    set_plot_style()
    figure_dir = output_root / "figures"
    data_dir = output_root / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    time_h = boundary["time_s"] / 3600.0
    data = pd.DataFrame(
        {
            "时间/h": time_h,
            "烘房温度/℃": boundary["air_temperature_c"],
            "外界水分浓度/(kg/kg)": boundary["air_moisture"],
        }
    )
    data.to_csv(data_dir / "figure_01_boundary_data.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(2, 1, figsize=(6.3, 5.4), sharex=True)
    axes[0].plot(time_h, boundary["air_temperature_c"], color="#0072B2", linewidth=1.5)
    axes[0].set_ylabel("烘房温度/℃")
    finish_axes(axes[0])
    axes[1].plot(time_h, boundary["air_moisture"], color="#D55E00", linewidth=1.5)
    axes[1].set_xlabel("时间/h")
    axes[1].set_ylabel("外界水分浓度/(kg/kg)")
    finish_axes(axes[1])
    figure.tight_layout(pad=1.0)
    figure.savefig(figure_dir / "figure_01_boundary.png", bbox_inches="tight")
    figure.savefig(figure_dir / "figure_01_boundary.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_q1_profiles(solution, output_root=Path("outputs")):
    set_plot_style()
    figure_dir = output_root / "figures"
    data_dir = output_root / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_time_s = np.arange(0.0, 1801.0, 300.0)
    radius_m = np.arange(0.0, 0.0200001, 0.001)
    temperature = interpolate_profiles(solution, selected_time_s, radius_m, "temperature_c")
    moisture = interpolate_profiles(solution, selected_time_s, radius_m, "moisture")

    rows = []
    for time_index, time_s in enumerate(selected_time_s):
        for radius_index, current_radius_m in enumerate(radius_m):
            rows.append(
                {
                    "时间/min": time_s / 60.0,
                    "到中心距离/cm": current_radius_m * 100.0,
                    "温度/℃": temperature[time_index, radius_index],
                    "水分浓度/(kg/kg)": moisture[time_index, radius_index],
                }
            )
    pd.DataFrame(rows).to_csv(
        data_dir / "figure_02_q1_profiles_data.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    colors = plt.colormaps["viridis"](np.linspace(0.08, 0.92, len(selected_time_s)))
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    for index, time_s in enumerate(selected_time_s):
        label = f"{time_s / 60:g} min"
        axes[0].plot(radius_m * 100.0, temperature[index], color=colors[index], linewidth=1.35, label=label)
        axes[1].plot(radius_m * 100.0, moisture[index], color=colors[index], linewidth=1.35, label=label)
    axes[0].set_xlabel("到药材中心的距离/cm")
    axes[0].set_ylabel("温度/℃")
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, va="top")
    axes[1].set_xlabel("到药材中心的距离/cm")
    axes[1].set_ylabel("水分浓度/(kg/kg)")
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, va="top")
    for axis in axes:
        finish_axes(axis)
        axis.set_xlim(0.0, 2.0)
    axes[1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=False, fontsize=8)
    figure.tight_layout(pad=0.8)
    figure.savefig(figure_dir / "figure_02_q1_profiles.png", bbox_inches="tight")
    figure.savefig(figure_dir / "figure_02_q1_profiles.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_q2_profiles(solution, output_root=Path("outputs")):
    set_plot_style()
    figure_dir = output_root / "figures"
    data_dir = output_root / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_time_s = np.arange(0.0, 10801.0, 1800.0)
    radius_m = np.arange(0.0, 0.0200001, 0.001)
    temperature = interpolate_profiles(solution, selected_time_s, radius_m, "temperature_c")
    moisture = interpolate_profiles(solution, selected_time_s, radius_m, "moisture")
    _, _, _, diffusivity = properties_q23(temperature, moisture)

    rows = []
    for time_index, time_s in enumerate(selected_time_s):
        for radius_index, current_radius_m in enumerate(radius_m):
            rows.append(
                {
                    "时间/h": time_s / 3600.0,
                    "到中心距离/cm": current_radius_m * 100.0,
                    "温度/℃": temperature[time_index, radius_index],
                    "水分浓度/(kg/kg)": moisture[time_index, radius_index],
                    "扩散系数/(m2/s)": diffusivity[time_index, radius_index],
                }
            )
    pd.DataFrame(rows).to_csv(
        data_dir / "figure_03_q2_profiles_data.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10e",
    )

    colors = plt.colormaps["viridis"](np.linspace(0.08, 0.92, len(selected_time_s)))
    figure = plt.figure(figsize=(7.1, 5.4))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.32)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    ]

    for index, time_s in enumerate(selected_time_s):
        label = f"{time_s / 3600:g} h"
        radius_cm = radius_m * 100.0
        axes[0].plot(radius_cm, temperature[index], color=colors[index], linewidth=1.35, label=label)
        axes[1].plot(radius_cm, moisture[index], color=colors[index], linewidth=1.35, label=label)
        axes[2].plot(radius_cm, diffusivity[index] * 1e9, color=colors[index], linewidth=1.35, label=label)

    axes[0].set_ylabel("温度/℃")
    axes[1].set_ylabel("水分浓度/(kg/kg)")
    axes[2].set_ylabel("扩散系数/(10⁻⁹ m²/s)")
    for label, axis in zip(["(a)", "(b)", "(c)"], axes):
        axis.set_xlabel("到药材中心的距离/cm")
        axis.set_xlim(0.0, 2.0)
        axis.text(0.02, 0.95, label, transform=axis.transAxes, va="top")
        finish_axes(axis)

    handles, labels = axes[2].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=7,
        frameon=False,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.14)
    figure.savefig(figure_dir / "figure_03_q2_profiles.png", bbox_inches="tight")
    figure.savefig(figure_dir / "figure_03_q2_profiles.pdf", bbox_inches="tight")
    plt.close(figure)
