from pathlib import Path

import numpy as np
import pandas as pd

from data_processing import boundary_values, interpolate_profiles
from model import (
    CP_Q1,
    H_M,
    H_T,
    INITIAL_MOISTURE,
    K_Q1,
    RHO_Q1,
    build_radial_grid,
    diffusivity_q1,
    harmonic_mean,
    q1_rhs,
    properties_q23,
    q2_rhs,
    solve_q1,
    solve_q2,
)


KEY_TIMES = np.array([100, 300, 600, 900, 1200, 1500, 1800], dtype=float)
KEY_RADIUS_M = np.array([0, 0.005, 0.010, 0.015, 0.020], dtype=float)


def thomas_solve(lower, diagonal, upper, right_hand_side):
    lower = lower.copy()
    diagonal = diagonal.copy()
    upper = upper.copy()
    right_hand_side = right_hand_side.copy()

    for index in range(1, len(diagonal)):
        factor = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= factor * upper[index - 1]
        right_hand_side[index] -= factor * right_hand_side[index - 1]

    solution = np.empty_like(right_hand_side)
    solution[-1] = right_hand_side[-1] / diagonal[-1]
    for index in range(len(diagonal) - 2, -1, -1):
        solution[index] = (
            right_hand_side[index] - upper[index] * solution[index + 1]
        ) / diagonal[index]
    return solution


def assemble_operator(coefficient, capacity, transfer_coefficient, external_value, grid):
    radius, _, volume, face_area = grid
    spacing = radius[1] - radius[0]
    node_count = len(radius)
    interface = harmonic_mean(coefficient[:-1], coefficient[1:])

    lower = np.zeros(node_count - 1)
    diagonal = np.zeros(node_count)
    upper = np.zeros(node_count - 1)
    source = np.zeros(node_count)

    for index in range(node_count):
        divisor = capacity * volume[index]
        if index > 0:
            conductance = face_area[index] * interface[index - 1] / spacing
            lower[index - 1] = conductance / divisor
            diagonal[index] -= conductance / divisor
        if index < node_count - 1:
            conductance = face_area[index + 1] * interface[index] / spacing
            upper[index] = conductance / divisor
            diagonal[index] -= conductance / divisor
        else:
            conductance = face_area[-1] * transfer_coefficient
            diagonal[index] -= conductance / divisor
            source[index] = conductance * external_value / divisor
    return lower, diagonal, upper, source


def apply_operator(state, operator):
    lower, diagonal, upper, source = operator
    result = diagonal * state + source
    result[1:] += lower * state[:-1]
    result[:-1] += upper * state[1:]
    return result


def crank_nicolson_q1(boundary, interval_count=80, time_step_s=1.0):
    grid = build_radial_grid(interval_count)
    node_count = len(grid[0])
    output_time = np.arange(0.0, 1800.0 + time_step_s, time_step_s)
    temperature = np.empty((len(output_time), node_count))
    moisture = np.empty((len(output_time), node_count))
    temperature[0] = 28.0
    moisture[0] = INITIAL_MOISTURE

    for time_index in range(len(output_time) - 1):
        time_old = output_time[time_index]
        time_new = output_time[time_index + 1]
        air_t_old, air_c_old = boundary_values(time_old, boundary)
        air_t_new, air_c_new = boundary_values(time_new, boundary)

        heat_old = assemble_operator(
            np.full(node_count, K_Q1), RHO_Q1 * CP_Q1, H_T, air_t_old, grid
        )
        heat_new = assemble_operator(
            np.full(node_count, K_Q1), RHO_Q1 * CP_Q1, H_T, air_t_new, grid
        )
        heat_rhs = temperature[time_index] + 0.5 * time_step_s * (
            apply_operator(temperature[time_index], heat_old) + heat_new[3]
        )
        lower, diagonal, upper, _ = heat_new
        temperature[time_index + 1] = thomas_solve(
            -0.5 * time_step_s * lower,
            1.0 - 0.5 * time_step_s * diagonal,
            -0.5 * time_step_s * upper,
            heat_rhs,
        )

        old_state = moisture[time_index]
        old_operator = assemble_operator(
            diffusivity_q1(old_state), 1.0, H_M, air_c_old, grid
        )
        old_rate_without_new_source = apply_operator(old_state, old_operator)
        iterate = old_state.copy()
        for _ in range(50):
            new_operator = assemble_operator(
                diffusivity_q1(iterate), 1.0, H_M, air_c_new, grid
            )
            moisture_rhs = old_state + 0.5 * time_step_s * (
                old_rate_without_new_source + new_operator[3]
            )
            lower, diagonal, upper, _ = new_operator
            updated = thomas_solve(
                -0.5 * time_step_s * lower,
                1.0 - 0.5 * time_step_s * diagonal,
                -0.5 * time_step_s * upper,
                moisture_rhs,
            )
            if np.max(np.abs(updated - iterate)) < 1e-11:
                iterate = updated
                break
            iterate = updated
        else:
            raise RuntimeError(f"CN–Picard在t={time_new:g} s未收敛")
        moisture[time_index + 1] = iterate

    return {
        "time_s": output_time,
        "radius_m": grid[0],
        "temperature_c": temperature,
        "moisture": moisture,
        "grid": grid,
    }


def cumulative_moisture_balance(solution, boundary):
    time_s = solution["time_s"]
    moisture = solution["moisture"]
    _, _, volume, face_area = solution["grid"]
    mean_moisture = moisture @ volume / volume.sum()
    _, air_moisture = boundary_values(time_s, boundary)
    outward_rate = face_area[-1] * H_M * (moisture[:, -1] - air_moisture) / volume.sum()
    cumulative = np.zeros_like(time_s)
    increments = 0.5 * np.diff(time_s) * (outward_rate[:-1] + outward_rate[1:])
    cumulative[1:] = np.cumsum(increments)
    residual = mean_moisture - mean_moisture[0] + cumulative
    relative_residual = np.max(np.abs(residual)) / max(
        abs(mean_moisture[0] - mean_moisture[-1]), 1e-12
    )
    return relative_residual, residual[-1]


def run_q1_validation(boundary, output_root=Path("outputs")):
    validation_dir = output_root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    solutions = {}
    for interval_count in [40, 80, 160, 320]:
        solutions[interval_count] = solve_q1(boundary, interval_count)

    finest = solutions[320]

    convergence_rows = []
    for field, unit in [("temperature_c", "℃"), ("moisture", "kg/kg")]:
        values_40 = interpolate_profiles(solutions[40], KEY_TIMES, KEY_RADIUS_M, field)
        values_80 = interpolate_profiles(solutions[80], KEY_TIMES, KEY_RADIUS_M, field)
        values_160 = interpolate_profiles(solutions[160], KEY_TIMES, KEY_RADIUS_M, field)
        values_320 = interpolate_profiles(solutions[320], KEY_TIMES, KEY_RADIUS_M, field)
        convergence_rows.append(
            {
                "field": field,
                "unit": unit,
                "max_difference_40_80": np.max(np.abs(values_40 - values_80)),
                "max_difference_80_160": np.max(np.abs(values_80 - values_160)),
                "max_difference_160_320": np.max(np.abs(values_160 - values_320)),
            }
        )
    convergence = pd.DataFrame(convergence_rows)

    cn_solution = crank_nicolson_q1(boundary, interval_count=80, time_step_s=1.0)
    comparison_rows = []
    for field, unit in [("temperature_c", "℃"), ("moisture", "kg/kg")]:
        bdf_values = interpolate_profiles(solutions[80], KEY_TIMES, KEY_RADIUS_M, field)
        cn_values = interpolate_profiles(cn_solution, KEY_TIMES, KEY_RADIUS_M, field)
        comparison_rows.append(
            {
                "field": field,
                "unit": unit,
                "max_abs_difference": np.max(np.abs(bdf_values - cn_values)),
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    relative_balance, final_balance = cumulative_moisture_balance(finest, boundary)

    temperature_radial_violation = np.min(np.diff(finest["temperature_c"], axis=1))
    moisture_radial_violation = np.max(np.diff(finest["moisture"], axis=1))
    alpha = K_Q1 / (RHO_Q1 * CP_Q1)
    initial_diffusivity = diffusivity_q1(np.array([INITIAL_MOISTURE]))[0]
    radius_m = finest["radius_m"][-1]
    dimensionless = pd.DataFrame(
        [
            {"quantity": "thermal_diffusivity", "value": alpha, "unit": "m2/s"},
            {"quantity": "initial_moisture_diffusivity", "value": initial_diffusivity, "unit": "m2/s"},
            {"quantity": "thermal_characteristic_time", "value": radius_m**2 / alpha / 3600, "unit": "h"},
            {"quantity": "moisture_characteristic_time", "value": radius_m**2 / initial_diffusivity / 3600, "unit": "h"},
            {"quantity": "thermal_Fourier_number_1800s", "value": alpha * 1800 / radius_m**2, "unit": "1"},
            {"quantity": "moisture_Fourier_number_1800s", "value": initial_diffusivity * 1800 / radius_m**2, "unit": "1"},
            {"quantity": "thermal_to_moisture_diffusivity_ratio", "value": alpha / initial_diffusivity, "unit": "1"},
        ]
    )

    uniform_grid = build_radial_grid(20)
    uniform_state = np.concatenate([np.full(21, 28.0), np.full(21, INITIAL_MOISTURE)])
    artificial_boundary = {
        "time_s": np.array([0.0, 60.0]),
        "air_temperature_c": np.array([28.0, 28.0]),
        "air_moisture": np.array([INITIAL_MOISTURE, INITIAL_MOISTURE]),
    }
    uniform_rate = q1_rhs(0.0, uniform_state, artificial_boundary, uniform_grid)

    summary = pd.DataFrame(
        [
            {"check": "uniform_field_rhs_max", "value": np.max(np.abs(uniform_rate)), "unit": "state/s"},
            {"check": "cumulative_moisture_balance_relative", "value": relative_balance, "unit": "1"},
            {"check": "cumulative_moisture_balance_final", "value": final_balance, "unit": "kg/kg"},
            {"check": "minimum_temperature", "value": finest["temperature_c"].min(), "unit": "℃"},
            {"check": "maximum_temperature", "value": finest["temperature_c"].max(), "unit": "℃"},
            {"check": "minimum_moisture", "value": finest["moisture"].min(), "unit": "kg/kg"},
            {"check": "maximum_moisture", "value": finest["moisture"].max(), "unit": "kg/kg"},
            {"check": "minimum_radial_temperature_increment", "value": temperature_radial_violation, "unit": "℃"},
            {"check": "maximum_radial_moisture_increment", "value": moisture_radial_violation, "unit": "kg/kg"},
        ]
    )

    temperature_grid_error = convergence.loc[
        convergence["field"] == "temperature_c", "max_difference_160_320"
    ].iloc[0]
    moisture_grid_error = convergence.loc[
        convergence["field"] == "moisture", "max_difference_160_320"
    ].iloc[0]
    if temperature_grid_error > 0.02 or moisture_grid_error > 5e-4:
        raise RuntimeError(
            "Q1网格验收未通过："
            f"温度差={temperature_grid_error:.3e} ℃，"
            f"含水率差={moisture_grid_error:.3e} kg/kg"
        )
    if temperature_radial_violation < -1e-8 or moisture_radial_violation > 1e-8:
        raise RuntimeError("Q1径向剖面的物理单调性检查未通过")

    convergence.to_csv(validation_dir / "q1_grid_convergence.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(validation_dir / "q1_bdf_cn_comparison.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(validation_dir / "q1_validation_summary.csv", index=False, encoding="utf-8-sig")
    dimensionless.to_csv(validation_dir / "q1_dimensionless_numbers.csv", index=False, encoding="utf-8-sig")
    return finest, convergence, comparison, summary


def run_q2_validation(boundary, output_root=Path("outputs")):
    validation_dir = output_root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    key_times = np.arange(0.0, 10801.0, 1800.0)

    solutions = {
        80: solve_q2(boundary, 80, key_times),
        160: solve_q2(boundary, 160, key_times),
        320: solve_q2(boundary, 320),
    }
    convergence_rows = []
    for field, unit in [("temperature_c", "℃"), ("moisture", "kg/kg")]:
        values_80 = interpolate_profiles(solutions[80], key_times, KEY_RADIUS_M, field)
        values_160 = interpolate_profiles(solutions[160], key_times, KEY_RADIUS_M, field)
        values_320 = interpolate_profiles(solutions[320], key_times, KEY_RADIUS_M, field)
        convergence_rows.append(
            {
                "field": field,
                "unit": unit,
                "max_difference_80_160": np.max(np.abs(values_80 - values_160)),
                "max_difference_160_320": np.max(np.abs(values_160 - values_320)),
            }
        )
    convergence = pd.DataFrame(convergence_rows)

    radau = solve_q2(boundary, 80, key_times, method="Radau")
    comparison_rows = []
    for field, unit in [("temperature_c", "℃"), ("moisture", "kg/kg")]:
        bdf_values = interpolate_profiles(solutions[80], key_times, KEY_RADIUS_M, field)
        radau_values = interpolate_profiles(radau, key_times, KEY_RADIUS_M, field)
        comparison_rows.append(
            {
                "field": field,
                "unit": unit,
                "max_abs_difference": np.max(np.abs(bdf_values - radau_values)),
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    finest = solutions[320]
    temperature_increment = np.min(np.diff(finest["temperature_c"], axis=1))
    moisture_increment = np.max(np.diff(finest["moisture"], axis=1))
    density, heat_capacity, conductivity, diffusivity = properties_q23(
        finest["temperature_c"],
        finest["moisture"],
    )

    uniform_grid = build_radial_grid(20)
    uniform_state = np.concatenate([np.full(21, 28.0), np.full(21, INITIAL_MOISTURE)])
    artificial_boundary = {
        "time_s": np.array([0.0, 10800.0]),
        "air_temperature_c": np.array([28.0, 28.0]),
        "air_moisture": np.array([INITIAL_MOISTURE, INITIAL_MOISTURE]),
    }
    uniform_rate = q2_rhs(0.0, uniform_state, artificial_boundary, uniform_grid)

    summary = pd.DataFrame(
        [
            {"check": "uniform_field_rhs_max", "value": np.max(np.abs(uniform_rate)), "unit": "state/s"},
            {"check": "minimum_temperature", "value": finest["temperature_c"].min(), "unit": "℃"},
            {"check": "maximum_temperature", "value": finest["temperature_c"].max(), "unit": "℃"},
            {"check": "minimum_moisture", "value": finest["moisture"].min(), "unit": "kg/kg"},
            {"check": "maximum_moisture", "value": finest["moisture"].max(), "unit": "kg/kg"},
            {"check": "minimum_density", "value": density.min(), "unit": "kg/m3"},
            {"check": "minimum_heat_capacity", "value": heat_capacity.min(), "unit": "J/(kg K)"},
            {"check": "minimum_conductivity", "value": conductivity.min(), "unit": "W/(m K)"},
            {"check": "minimum_diffusivity", "value": diffusivity.min(), "unit": "m2/s"},
            {"check": "maximum_diffusivity", "value": diffusivity.max(), "unit": "m2/s"},
            {"check": "minimum_radial_temperature_increment", "value": temperature_increment, "unit": "℃"},
            {"check": "maximum_radial_moisture_increment", "value": moisture_increment, "unit": "kg/kg"},
        ]
    )

    temperature_error = convergence.loc[
        convergence["field"] == "temperature_c", "max_difference_160_320"
    ].iloc[0]
    moisture_error = convergence.loc[
        convergence["field"] == "moisture", "max_difference_160_320"
    ].iloc[0]
    if temperature_error > 0.02 or moisture_error > 5e-4:
        raise RuntimeError(
            "Q2网格验收未通过："
            f"温度差={temperature_error:.3e} ℃，"
            f"含水率差={moisture_error:.3e} kg/kg"
        )
    air_temperature_min = min(28.0, boundary["air_temperature_c"].min())
    air_temperature_max = max(28.0, boundary["air_temperature_c"].max())
    if finest["temperature_c"].min() < air_temperature_min - 1e-6:
        raise RuntimeError("Q2温度低于初始值与烘房边界的共同下界")
    if finest["temperature_c"].max() > air_temperature_max + 1e-6:
        raise RuntimeError("Q2温度高于初始值与烘房边界的共同上界")
    if moisture_increment > 1e-7:
        raise RuntimeError("Q2含水率径向单调性检查未通过")

    convergence.to_csv(validation_dir / "q2_grid_convergence.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(validation_dir / "q2_bdf_radau_comparison.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(validation_dir / "q2_validation_summary.csv", index=False, encoding="utf-8-sig")
    return finest, convergence, comparison, summary
