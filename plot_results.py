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


def plot_q3_endpoint(solution, sensitivity, output_root=Path("outputs")):
    set_plot_style()
    figure_dir = output_root / "figures"
    data_dir = output_root / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    time_h = solution["time_s"] / 3600.0
    maximum_moisture = np.max(solution["moisture"], axis=1)
    data = pd.DataFrame(
        {
            "时间/h": time_h,
            "全域最大水分浓度/(kg/kg)": maximum_moisture,
            "中心水分浓度/(kg/kg)": solution["moisture"][:, 0],
            "表面水分浓度/(kg/kg)": solution["moisture"][:, -1],
        }
    )
    data.to_csv(
        data_dir / "figure_04_q3_endpoint_data.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    axes[0].plot(time_h, maximum_moisture, color="#0072B2", linewidth=1.5, label="全域最大值")
    axes[0].plot(time_h, solution["moisture"][:, -1], color="#D55E00", linewidth=1.2, linestyle="--", label="表面")
    axes[0].axhline(0.15, color="#000000", linewidth=1.0, linestyle=":", label="达标阈值")
    axes[0].axvline(solution["endpoint_s"] / 3600.0, color="#009E73", linewidth=1.0, linestyle="-.")
    axes[0].set_xlabel("时间/h")
    axes[0].set_ylabel("水分浓度/(kg/kg)")
    axes[0].legend(frameon=False, fontsize=8)

    endpoint_h = solution["endpoint_s"] / 3600.0
    near = np.abs(time_h - endpoint_h) <= 3.0
    axes[1].plot(time_h[near], maximum_moisture[near], color="#0072B2", linewidth=1.5)
    axes[1].axhline(0.15, color="#000000", linewidth=1.0, linestyle=":")
    axes[1].axvline(endpoint_h, color="#009E73", linewidth=1.0, linestyle="-.", label=f"{endpoint_h:.4f} h")
    axes[1].scatter([endpoint_h], [0.15], color="#009E73", s=22, zorder=3)
    axes[1].set_xlabel("时间/h")
    axes[1].set_ylabel("全域最大水分浓度/(kg/kg)")
    axes[1].legend(frameon=False, fontsize=8)
    for label, axis in zip(["(a)", "(b)"], axes):
        axis.text(0.02, 0.96, label, transform=axis.transAxes, va="top")
        finish_axes(axis)
    figure.tight_layout(pad=0.8)
    figure.savefig(figure_dir / "figure_04_q3_endpoint.png", bbox_inches="tight")
    figure.savefig(figure_dir / "figure_04_q3_endpoint.pdf", bbox_inches="tight")
    plt.close(figure)

    grid = sensitivity[sensitivity["scenario"] == "D_h_grid"].copy()
    single_d = grid[grid["mass_transfer_scale"] == 1.0]
    single_h = grid[grid["diffusivity_scale"] == 1.0]
    figure, axis = plt.subplots(figsize=(6.3, 3.5))
    axis.plot(
        single_d["diffusivity_scale"],
        single_d["endpoint_h"],
        color="#0072B2",
        marker="o",
        linewidth=1.4,
        label="扩散系数倍率",
    )
    axis.plot(
        single_h["mass_transfer_scale"],
        single_h["endpoint_h"],
        color="#D55E00",
        marker="s",
        linestyle="--",
        linewidth=1.4,
        label="表面对流传质倍率",
    )
    axis.set_xlabel("参数倍率")
    axis.set_ylabel("烘干终点/h")
    axis.set_xticks([0.9, 1.0, 1.1])
    axis.legend(frameon=False)
    finish_axes(axis)
    figure.tight_layout(pad=0.8)
    figure.savefig(figure_dir / "figure_05_q3_sensitivity.png", bbox_inches="tight")
    figure.savefig(figure_dir / "figure_05_q3_sensitivity.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_q4_analysis(
    solution, radius_history, decomposition, sensitivity, output_root=Path("outputs")
):
    set_plot_style()
    figure_dir = output_root / "figures"
    data_dir = output_root / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    radius_data = pd.DataFrame(
        {
            "时间/h": solution["time_s"] / 3600.0,
            "药材半径/cm": solution["radius_at_time_m"] * 100.0,
        }
    )
    radius_data.to_csv(
        data_dir / "figure_06_q4_radius_data.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    figure = plt.figure(figsize=(7.1, 5.2))
    grid = figure.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    axes = [
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]
    axes[0].plot(
        radius_history["time_s"] / 3600.0,
        radius_history["radius_m"] * 100.0,
        color="#000000",
        marker="o",
        markersize=2.5,
        linestyle="none",
        label="附件2",
    )
    axes[0].plot(
        radius_data["时间/h"],
        radius_data["药材半径/cm"],
        color="#0072B2",
        linewidth=1.4,
        label="PCHIP",
    )
    axes[0].set_xlabel("时间/h")
    axes[0].set_ylabel("药材半径/cm")
    axes[0].legend(frameon=False, ncol=2)

    labels = ["P3物性\n固定半径", "P4物性\n固定半径", "P4物性\n收缩半径"]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    axes[1].bar(labels, decomposition["endpoint_h"], color=colors, width=0.65)
    axes[1].set_ylabel("烘干终点/h")
    axes[1].tick_params(axis="x", labelsize=8)

    shrinkage = sensitivity[sensitivity["scenario"] == "shrinkage_scale"]
    axes[2].plot(
        shrinkage["shrinkage_scale"],
        shrinkage["endpoint_h"],
        color="#CC79A7",
        marker="D",
        linewidth=1.4,
    )
    axes[2].set_xlabel("收缩幅度倍率")
    axes[2].set_ylabel("烘干终点/h")
    axes[2].set_xticks([0.9, 1.0, 1.1])
    for label, axis in zip(["(a)", "(b)", "(c)"], axes):
        axis.text(0.02, 0.96, label, transform=axis.transAxes, va="top")
        finish_axes(axis)
    figure.savefig(figure_dir / "figure_06_q4_effects.png", bbox_inches="tight")
    figure.savefig(figure_dir / "figure_06_q4_effects.pdf", bbox_inches="tight")
    plt.close(figure)

    selected_time_s = np.arange(0.0, solution["strict_output_s"], 43200.0)
    selected_time_s = np.append(selected_time_s, solution["strict_output_s"])
    selected_indices = np.searchsorted(solution["time_s"], selected_time_s)
    rows = []
    for time_s, index in zip(selected_time_s, selected_indices):
        physical_radius_cm = (
            solution["material_coordinate"] * solution["radius_at_time_m"][index] * 100.0
        )
        for radius_cm, temperature, moisture in zip(
            physical_radius_cm,
            solution["temperature_c"][index],
            solution["moisture"][index],
        ):
            rows.append(
                {
                    "时间/h": time_s / 3600.0,
                    "到中心距离/cm": radius_cm,
                    "温度/℃": temperature,
                    "水分浓度/(kg/kg)": moisture,
                }
            )
    pd.DataFrame(rows).to_csv(
        data_dir / "figure_07_q4_profiles_data.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    colors = plt.colormaps["viridis"](np.linspace(0.08, 0.92, len(selected_time_s)))
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    for plot_index, (time_s, index) in enumerate(zip(selected_time_s, selected_indices)):
        radius_cm = (
            solution["material_coordinate"] * solution["radius_at_time_m"][index] * 100.0
        )
        label = f"{time_s / 3600:g} h"
        axes[0].plot(radius_cm, solution["temperature_c"][index], color=colors[plot_index], linewidth=1.35, label=label)
        axes[1].plot(radius_cm, solution["moisture"][index], color=colors[plot_index], linewidth=1.35, label=label)
    axes[0].set_xlabel("到药材中心的距离/cm")
    axes[0].set_ylabel("温度/℃")
    axes[1].set_xlabel("到药材中心的距离/cm")
    axes[1].set_ylabel("水分浓度/(kg/kg)")
    for label, axis in zip(["(a)", "(b)"], axes):
        axis.set_xlim(0.0, 2.0)
        axis.text(0.02, 0.96, label, transform=axis.transAxes, va="top")
        finish_axes(axis)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    figure.tight_layout(pad=0.8)
    figure.savefig(figure_dir / "figure_07_q4_profiles.png", bbox_inches="tight")
    figure.savefig(figure_dir / "figure_07_q4_profiles.pdf", bbox_inches="tight")
    plt.close(figure)
