from pathlib import Path

import numpy as np
import pandas as pd


BOUNDARY_FILE = Path("附件/附件1.xlsx")


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


def boundary_values(time_s, boundary):
    if np.any(np.asarray(time_s) < 0):
        raise ValueError("时间不能为负")

    measured_time = boundary["time_s"]
    temperature = np.interp(
        time_s,
        measured_time,
        boundary["air_temperature_c"],
        right=50.0,
    )
    moisture = np.interp(
        time_s,
        measured_time,
        boundary["air_moisture"],
        right=0.05,
    )
    return temperature, moisture


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
