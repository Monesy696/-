from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


BOUNDARY_FILE = Path("附件/附件1.xlsx")
RADIUS_FILE = Path("附件/附件2.xlsx")


def load_boundary(path=BOUNDARY_FILE):
    data = pd.read_excel(path)
    required = ["时间", "温度", "水分浓度"]
    if list(data.columns[:3]) != required:
        raise ValueError(f"附件1前三列应为{required}，实际为{list(data.columns[:3])}")
    if data[required].isna().any().any():
        raise ValueError("附件1存在缺失值")

    time_s = data["时间"].to_numpy(dtype=float)
    air_temperature_c = data["温度"].to_numpy(dtype=float)
    air_moisture = data["水分浓度"].to_numpy(dtype=float)

    if time_s[0] != 0 or time_s[-1] != 14400:
        raise ValueError("附件1时间范围应为0–14400 s")
    if not np.allclose(np.diff(time_s), 60.0):
        raise ValueError("附件1时间间隔应为60 s")
    if np.any(air_temperature_c < -273.15) or np.any(air_moisture < 0):
        raise ValueError("附件1存在不合理的温度或水分边界")

    return {
        "time_s": time_s,
        "air_temperature_c": air_temperature_c,
        "air_moisture": air_moisture,
    }


def boundary_values(time_s, boundary, long_term_mode="rounded"):
    if np.any(np.asarray(time_s) < 0):
        raise ValueError("时间不能为负")
    if long_term_mode not in {"rounded", "last"}:
        raise ValueError("长期边界模式必须为rounded或last")

    measured_time = boundary["time_s"]
    if long_term_mode == "rounded":
        terminal_temperature = 50.0
        terminal_moisture = 0.05
    else:
        terminal_temperature = boundary["air_temperature_c"][-1]
        terminal_moisture = boundary["air_moisture"][-1]
    temperature = np.interp(
        time_s,
        measured_time,
        boundary["air_temperature_c"],
        right=terminal_temperature,
    )
    moisture = np.interp(
        time_s,
        measured_time,
        boundary["air_moisture"],
        right=terminal_moisture,
    )
    return temperature, moisture


def load_radius_history(path=RADIUS_FILE):
    data = pd.read_excel(path)
    required = ["时间", "半径"]
    if list(data.columns[:2]) != required:
        raise ValueError(f"附件2前两列应为{required}，实际为{list(data.columns[:2])}")
    if data[required].isna().any().any():
        raise ValueError("附件2存在缺失值")

    time_s = data["时间"].to_numpy(dtype=float)
    radius_m = data["半径"].to_numpy(dtype=float) / 100.0
    if time_s[0] != 0 or not np.allclose(np.diff(time_s), 1800.0):
        raise ValueError("附件2时间必须从0 s开始且间隔为1800 s")
    if np.any(radius_m <= 0) or np.any(np.diff(radius_m) > 1e-12):
        raise ValueError("附件2半径必须为正且非增")

    return {
        "time_s": time_s,
        "radius_m": radius_m,
        "pchip": PchipInterpolator(time_s, radius_m, extrapolate=False),
    }


def radius_values(time_s, radius_history, method="pchip", shrinkage_scale=1.0):
    if method not in {"pchip", "linear"}:
        raise ValueError("半径插值方法必须为pchip或linear")
    if shrinkage_scale <= 0:
        raise ValueError("收缩幅度参数必须为正")

    requested_time = np.asarray(time_s, dtype=float)
    if np.any(requested_time < 0):
        raise ValueError("时间不能为负")
    source_time = radius_history["time_s"]
    source_radius = radius_history["radius_m"]
    clipped_time = np.minimum(requested_time, source_time[-1])
    if method == "pchip":
        measured_radius = np.asarray(radius_history["pchip"](clipped_time), dtype=float)
    else:
        measured_radius = np.interp(clipped_time, source_time, source_radius)

    initial_radius = source_radius[0]
    radius = initial_radius - shrinkage_scale * (initial_radius - measured_radius)
    if np.any(radius <= 0):
        raise ValueError("收缩幅度导致半径非正")
    return float(radius) if radius.ndim == 0 else radius


def export_q1(solution, output_root=Path("outputs")):
    output_root.mkdir(parents=True, exist_ok=True)
    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    output_time = np.arange(1.0, 1801.0, 1.0)
    output_radius_m = np.arange(0.0, 0.0200001, 0.001)
    temperature = interpolate_profiles(solution, output_time, output_radius_m, "temperature_c")
    moisture = interpolate_profiles(solution, output_time, output_radius_m, "moisture")

    radius_columns_cm = np.round(output_radius_m * 100, 1)
    temperature_frame = pd.DataFrame(temperature, columns=radius_columns_cm)
    moisture_frame = pd.DataFrame(moisture, columns=radius_columns_cm)
    time_header = "时间\\到药材中心的距离"
    temperature_frame.insert(0, time_header, output_time.astype(int))
    moisture_frame.insert(0, time_header, output_time.astype(int))

    result_path = output_root / "result1.xlsx"
    with pd.ExcelWriter(result_path, engine="openpyxl") as writer:
        temperature_frame.round(4).to_excel(writer, sheet_name="温度", index=False)
        moisture_frame.round(4).to_excel(writer, sheet_name="水分浓度", index=False)
        for sheet_name in ["温度", "水分浓度"]:
            sheet = writer.book[sheet_name]
            sheet.freeze_panes = "B2"
            sheet.column_dimensions["A"].width = 28
            for column_cells in sheet.iter_cols(min_col=2, max_col=sheet.max_column):
                sheet.column_dimensions[column_cells[0].column_letter].width = 11
                for cell in column_cells[1:]:
                    cell.number_format = "0.0000"
            for cell in sheet["A"][1:]:
                cell.number_format = "0"

    key_times = np.array([100, 300, 600, 900, 1200, 1500, 1800], dtype=float)
    key_radius_m = np.array([0, 0.005, 0.010, 0.015, 0.020], dtype=float)
    key_temperature = interpolate_profiles(solution, key_times, key_radius_m, "temperature_c")
    key_moisture = interpolate_profiles(solution, key_times, key_radius_m, "moisture")

    key_columns = ["时间/s"] + [f"r={radius_m * 100:g} cm" for radius_m in key_radius_m]
    table_temperature = pd.DataFrame(
        np.column_stack([key_times.astype(int), key_temperature]),
        columns=key_columns,
    )
    table_moisture = pd.DataFrame(
        np.column_stack([key_times.astype(int), key_moisture]),
        columns=key_columns,
    )
    table_temperature.to_csv(
        table_dir / "table_01_q1_temperature.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    table_moisture.to_csv(
        table_dir / "table_02_q1_moisture.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    return result_path, table_temperature, table_moisture


def export_q2(solution, output_root=Path("outputs")):
    output_root.mkdir(parents=True, exist_ok=True)
    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    output_time = np.arange(1.0, 10801.0, 1.0)
    output_radius_m = np.arange(0.0, 0.0200001, 0.001)
    temperature = interpolate_profiles(solution, output_time, output_radius_m, "temperature_c")
    moisture = interpolate_profiles(solution, output_time, output_radius_m, "moisture")

    radius_columns_cm = np.round(output_radius_m * 100, 1)
    temperature_frame = pd.DataFrame(temperature, columns=radius_columns_cm)
    moisture_frame = pd.DataFrame(moisture, columns=radius_columns_cm)
    time_header = "时间\\到药材中心的距离"
    temperature_frame.insert(0, time_header, output_time.astype(int))
    moisture_frame.insert(0, time_header, output_time.astype(int))

    result_path = output_root / "result2.xlsx"
    with pd.ExcelWriter(result_path, engine="openpyxl") as writer:
        temperature_frame.round(4).to_excel(writer, sheet_name="温度", index=False)
        moisture_frame.round(4).to_excel(writer, sheet_name="水分浓度", index=False)
        for sheet_name in ["温度", "水分浓度"]:
            sheet = writer.book[sheet_name]
            sheet.freeze_panes = "B2"
            sheet.column_dimensions["A"].width = 28
            for column_cells in sheet.iter_cols(min_col=2, max_col=sheet.max_column):
                sheet.column_dimensions[column_cells[0].column_letter].width = 11
                for cell in column_cells[1:]:
                    cell.number_format = "0.0000"
            for cell in sheet["A"][1:]:
                cell.number_format = "0"

    key_times = np.arange(1800.0, 10801.0, 1800.0)
    key_radius_m = np.array([0, 0.005, 0.010, 0.015, 0.020], dtype=float)
    key_temperature = interpolate_profiles(solution, key_times, key_radius_m, "temperature_c")
    key_moisture = interpolate_profiles(solution, key_times, key_radius_m, "moisture")
    key_columns = ["时间/h"] + [f"r={radius_m * 100:g} cm" for radius_m in key_radius_m]
    table_temperature = pd.DataFrame(
        np.column_stack([key_times / 3600.0, key_temperature]),
        columns=key_columns,
    )
    table_moisture = pd.DataFrame(
        np.column_stack([key_times / 3600.0, key_moisture]),
        columns=key_columns,
    )
    table_temperature.to_csv(
        table_dir / "table_03_q2_temperature.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    table_moisture.to_csv(
        table_dir / "table_04_q2_moisture.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    return result_path, table_temperature, table_moisture


def interpolate_profiles(solution, target_time_s, target_radius_m, field):
    source_time = solution["time_s"]
    source_radius = solution["radius_m"]
    source_values = solution[field]

    time_indices = np.searchsorted(source_time, target_time_s)
    if np.any(time_indices >= len(source_time)) or not np.allclose(
        source_time[time_indices], target_time_s
    ):
        raise ValueError("目标时间不在已求解的输出时刻中")

    sampled = np.empty((len(target_time_s), len(target_radius_m)))
    for row, time_index in enumerate(time_indices):
        sampled[row] = np.interp(
            target_radius_m,
            source_radius,
            source_values[time_index],
        )
    return sampled


def _write_single_sheet_result(frame, result_path):
    with pd.ExcelWriter(result_path, engine="openpyxl") as writer:
        frame.round(4).to_excel(writer, sheet_name="Sheet1", index=False)
        sheet = writer.book["Sheet1"]
        sheet.freeze_panes = "B2"
        sheet.column_dimensions["A"].width = 28
        for column_cells in sheet.iter_cols(min_col=2, max_col=sheet.max_column):
            sheet.column_dimensions[column_cells[0].column_letter].width = 11
            for cell in column_cells[1:]:
                cell.number_format = "0.0000"
        for cell in sheet["A"][1:]:
            cell.number_format = "0"


def _table_times(endpoint_s):
    regular = np.arange(21600.0, endpoint_s, 21600.0)
    return np.concatenate([regular, np.array([endpoint_s])])


def export_q3(solution, output_root=Path("outputs")):
    output_root.mkdir(parents=True, exist_ok=True)
    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    output_time = solution["time_s"][1:]
    output_radius_m = np.arange(0.0, 0.0200001, 0.001)
    moisture = interpolate_profiles(solution, output_time, output_radius_m, "moisture")
    frame = pd.DataFrame(moisture, columns=np.round(output_radius_m * 100.0, 1))
    frame.insert(0, "时间\\到药材中心的距离", output_time.astype(int))
    result_path = output_root / "result3.xlsx"
    _write_single_sheet_result(frame, result_path)

    table_radius_m = np.array([0.0, 0.005, 0.010, 0.015, 0.020])
    table_time_s = _table_times(solution["endpoint_s"])
    regular_time_s = table_time_s[:-1]
    if len(regular_time_s):
        regular_values = interpolate_profiles(
            solution, regular_time_s, table_radius_m, "moisture"
        )
    else:
        regular_values = np.empty((0, len(table_radius_m)))
    endpoint_values = np.interp(
        table_radius_m,
        solution["radius_m"],
        solution["endpoint_moisture"],
    )
    values = np.vstack([regular_values, endpoint_values])
    columns = ["时间/h"] + [f"r={radius * 100:g} cm" for radius in table_radius_m]
    table = pd.DataFrame(np.column_stack([table_time_s / 3600.0, values]), columns=columns)
    table.to_csv(
        table_dir / "table_05_q3_moisture.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    return result_path, table


def interpolate_shrinking_profiles(solution, target_time_s, target_radius_m, field):
    source_time = solution["time_s"]
    source_values = solution[field]
    material_coordinate = solution["material_coordinate"]
    radii = solution["radius_at_time_m"]
    target_time_s = np.asarray(target_time_s, dtype=float)
    target_radius_m = np.asarray(target_radius_m, dtype=float)
    time_indices = np.searchsorted(source_time, target_time_s)
    if np.any(time_indices >= len(source_time)) or not np.allclose(
        source_time[time_indices], target_time_s
    ):
        raise ValueError("目标时间不在已求解的输出时刻中")

    sampled = np.full((len(target_time_s), len(target_radius_m)), np.nan)
    for row, time_index in enumerate(time_indices):
        current_radius = radii[time_index]
        inside = target_radius_m <= current_radius + 1e-12
        xi = target_radius_m[inside] / current_radius
        sampled[row, inside] = np.interp(
            xi,
            material_coordinate,
            source_values[time_index],
        )
    return sampled


def _sample_shrinking_endpoint(solution, target_radius_m, field):
    current_radius = solution["endpoint_radius_m"]
    target_radius_m = np.asarray(target_radius_m, dtype=float)
    sampled = np.full(len(target_radius_m), np.nan)
    inside = target_radius_m <= current_radius + 1e-12
    sampled[inside] = np.interp(
        target_radius_m[inside] / current_radius,
        solution["material_coordinate"],
        solution[f"endpoint_{field}"],
    )
    return sampled


def export_q4(solution, output_root=Path("outputs")):
    output_root.mkdir(parents=True, exist_ok=True)
    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    output_time = solution["time_s"][1:]
    fixed_radius_m = np.arange(0.0, 0.0200001, 0.001)
    moisture = interpolate_shrinking_profiles(
        solution, output_time, fixed_radius_m, "moisture"
    )
    surface = solution["moisture"][1:, -1, None]
    frame = pd.DataFrame(
        np.column_stack([moisture, surface]),
        columns=[*np.round(fixed_radius_m * 100.0, 1), "药材表面"],
    )
    frame.insert(0, "时间\\到药材中心的距离", output_time.astype(int))
    result_path = output_root / "result4.xlsx"
    _write_single_sheet_result(frame, result_path)

    table_radius_m = np.array([0.0, 0.005, 0.010, 0.015, 0.020])
    table_time_s = _table_times(solution["endpoint_s"])
    regular_time_s = table_time_s[:-1]
    if len(regular_time_s):
        regular_values = interpolate_shrinking_profiles(
            solution, regular_time_s, table_radius_m, "moisture"
        )
        regular_indices = np.searchsorted(solution["time_s"], regular_time_s)
        regular_surface = solution["moisture"][regular_indices, -1, None]
    else:
        regular_values = np.empty((0, len(table_radius_m)))
        regular_surface = np.empty((0, 1))
    endpoint_values = _sample_shrinking_endpoint(solution, table_radius_m, "moisture")
    endpoint_surface = solution["endpoint_moisture"][-1]
    values = np.vstack(
        [
            np.column_stack([regular_values, regular_surface]),
            np.append(endpoint_values, endpoint_surface),
        ]
    )
    columns = [
        "时间/h",
        *[f"r={radius * 100:g} cm" for radius in table_radius_m],
        "药材表面",
    ]
    table = pd.DataFrame(np.column_stack([table_time_s / 3600.0, values]), columns=columns)
    table.to_csv(
        table_dir / "table_06_q4_moisture.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    return result_path, table
