from pathlib import Path

import numpy as np
import pandas as pd

from data_processing import boundary_values, interpolate_profiles, radius_values
from model import (
    CP_Q1,
    H_M,
    H_T,
    INITIAL_MOISTURE,
    K_Q1,
    LENGTH_M,
    RADIUS_M,
    RHO_Q1,
    build_radial_grid,
    diffusivity_q1,
    harmonic_mean,
    q1_rhs,
    q4_rhs,
    radial_rate,
    scale_material_grid,
    properties_q23,
    properties_q4,
    q2_rhs,
    solve_q1,
    solve_q2,
    solve_q3,
    solve_q3_endpoint,
    solve_q4,
    solve_q4_endpoint,
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
    spacing = np.diff(radius)
    node_count = len(radius)
    interface = harmonic_mean(coefficient[:-1], coefficient[1:])

    lower = np.zeros(node_count - 1)
    diagonal = np.zeros(node_count)
    upper = np.zeros(node_count - 1)
    source = np.zeros(node_count)

    for index in range(node_count):
        divisor = capacity * volume[index]
        if index > 0:
            conductance = face_area[index] * interface[index - 1] / spacing[index - 1]
            lower[index - 1] = conductance / divisor
            diagonal[index] -= conductance / divisor
        if index < node_count - 1:
            conductance = face_area[index + 1] * interface[index] / spacing[index]
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


def cumulative_fixed_moisture_balance(solution, boundary):
    time_s = solution["time_s"]
    moisture = solution["moisture"]
    _, _, volume, face_area = solution["grid"]
    mean_moisture = moisture @ volume / volume.sum()
    _, air_moisture = boundary_values(
        time_s, boundary, long_term_mode=solution["long_term_mode"]
    )
    outward_rate = (
        face_area[-1]
        * H_M
        * solution["mass_transfer_scale"]
        * (moisture[:, -1] - air_moisture)
        / volume.sum()
    )
    cumulative = np.zeros_like(time_s)
    cumulative[1:] = np.cumsum(
        0.5 * np.diff(time_s) * (outward_rate[:-1] + outward_rate[1:])
    )
    residual = mean_moisture - mean_moisture[0] + cumulative
    relative = np.max(np.abs(residual)) / max(
        abs(mean_moisture[0] - mean_moisture[-1]), 1e-12
    )
    return relative, residual[-1]


def _endpoint_grid_frame(endpoints):
    rows = []
    previous = None
    for interval_count in sorted(endpoints):
        endpoint_s = endpoints[interval_count]["endpoint_s"]
        rows.append(
            {
                "interval_count": interval_count,
                "endpoint_s": endpoint_s,
                "endpoint_h": endpoint_s / 3600.0,
                "difference_from_previous_s": (
                    np.nan if previous is None else endpoint_s - previous
                ),
            }
        )
        previous = endpoint_s
    return pd.DataFrame(rows)


def run_q3_validation(boundary, output_root=Path("outputs")):
    validation_dir = output_root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    endpoints = {
        interval_count: solve_q3_endpoint(
            boundary, interval_count=interval_count, max_step_s=300.0
        )
        for interval_count in [160, 320, 640]
    }
    convergence = _endpoint_grid_frame(endpoints)
    finest_endpoint = endpoints[640]
    solution = solve_q3(boundary, interval_count=640, endpoint=finest_endpoint)

    radau = solve_q3_endpoint(
        boundary, interval_count=160, method="Radau", max_step_s=300.0
    )
    comparison = pd.DataFrame(
        [
            {
                "interval_count": 160,
                "primary_method": "BDF",
                "check_method": "Radau",
                "bdf_endpoint_s": endpoints[160]["endpoint_s"],
                "radau_endpoint_s": radau["endpoint_s"],
                "absolute_difference_s": abs(
                    endpoints[160]["endpoint_s"] - radau["endpoint_s"]
                ),
            }
        ]
    )
    tightened = solve_q3_endpoint(
        boundary,
        interval_count=640,
        max_step_s=150.0,
        relative_tolerance=1e-8,
        temperature_tolerance=1e-8,
        moisture_tolerance=1e-10,
    )
    time_refinement = pd.DataFrame(
        [
            {
                "interval_count": 640,
                "baseline_endpoint_s": finest_endpoint["endpoint_s"],
                "tightened_endpoint_s": tightened["endpoint_s"],
                "absolute_difference_s": abs(
                    finest_endpoint["endpoint_s"] - tightened["endpoint_s"]
                ),
            }
        ]
    )

    sensitivity_rows = []
    scenario_cache = {(1.0, 1.0): endpoints[160]}
    for diffusivity_scale in [0.9, 1.0, 1.1]:
        for mass_transfer_scale in [0.9, 1.0, 1.1]:
            key = (diffusivity_scale, mass_transfer_scale)
            if key not in scenario_cache:
                scenario_cache[key] = solve_q3_endpoint(
                    boundary,
                    interval_count=160,
                    max_step_s=300.0,
                    diffusivity_scale=diffusivity_scale,
                    mass_transfer_scale=mass_transfer_scale,
                )
            endpoint_s = scenario_cache[key]["endpoint_s"]
            sensitivity_rows.append(
                {
                    "scenario": "D_h_grid",
                    "diffusivity_scale": diffusivity_scale,
                    "mass_transfer_scale": mass_transfer_scale,
                    "long_term_boundary": "rounded_50_0.05",
                    "endpoint_s": endpoint_s,
                    "endpoint_h": endpoint_s / 3600.0,
                }
            )
    last_boundary = solve_q3_endpoint(
        boundary,
        interval_count=160,
        max_step_s=300.0,
        long_term_mode="last",
    )
    sensitivity_rows.append(
        {
            "scenario": "last_boundary",
            "diffusivity_scale": 1.0,
            "mass_transfer_scale": 1.0,
            "long_term_boundary": "hold_attachment_last",
            "endpoint_s": last_boundary["endpoint_s"],
            "endpoint_h": last_boundary["endpoint_s"] / 3600.0,
        }
    )
    sensitivity = pd.DataFrame(sensitivity_rows)
    baseline_s = endpoints[160]["endpoint_s"]
    sensitivity["delta_from_baseline_h"] = (
        sensitivity["endpoint_s"] - baseline_s
    ) / 3600.0
    d_elasticity = (
        scenario_cache[(1.1, 1.0)]["endpoint_s"]
        - scenario_cache[(0.9, 1.0)]["endpoint_s"]
    ) / (0.2 * baseline_s)
    h_elasticity = (
        scenario_cache[(1.0, 1.1)]["endpoint_s"]
        - scenario_cache[(1.0, 0.9)]["endpoint_s"]
    ) / (0.2 * baseline_s)
    sensitivity_summary = pd.DataFrame(
        [
            {"parameter": "diffusivity_scale", "local_elasticity": d_elasticity},
            {"parameter": "mass_transfer_scale", "local_elasticity": h_elasticity},
        ]
    )

    balance_relative, balance_final = cumulative_fixed_moisture_balance(
        solution, boundary
    )
    max_before = np.max(solution["moisture"][-2])
    max_strict = np.max(solution["moisture"][-1])
    endpoint_index = int(np.argmax(solution["endpoint_moisture"]))
    radial_increment = np.max(np.diff(solution["moisture"], axis=1))
    summary = pd.DataFrame(
        [
            {"check": "continuous_endpoint", "value": solution["endpoint_s"], "unit": "s"},
            {"check": "continuous_endpoint", "value": solution["endpoint_s"] / 3600.0, "unit": "h"},
            {"check": "strict_60s_endpoint", "value": solution["strict_output_s"], "unit": "s"},
            {"check": "maximum_moisture_before_strict", "value": max_before, "unit": "kg/kg"},
            {"check": "maximum_moisture_at_strict", "value": max_strict, "unit": "kg/kg"},
            {"check": "endpoint_control_radius", "value": solution["radius_m"][endpoint_index] * 100.0, "unit": "cm"},
            {"check": "maximum_radial_moisture_increment", "value": radial_increment, "unit": "kg/kg"},
            {"check": "cumulative_moisture_balance_relative", "value": balance_relative, "unit": "1"},
            {"check": "cumulative_moisture_balance_final", "value": balance_final, "unit": "kg/kg"},
        ]
    )

    finest_grid_difference = abs(
        endpoints[640]["endpoint_s"] - endpoints[320]["endpoint_s"]
    )
    if finest_grid_difference > 30.0:
        raise RuntimeError(
            f"Q3终点网格验收未通过：320到640区间差{finest_grid_difference:.3f} s"
        )
    if time_refinement["absolute_difference_s"].iloc[0] > 5.0:
        raise RuntimeError("Q3时间积分收紧验收未通过")
    if comparison["absolute_difference_s"].iloc[0] > 5.0:
        raise RuntimeError("Q3 BDF与Radau终点复核未通过")
    if max_before < 0.15 or max_strict >= 0.15:
        raise RuntimeError("Q3首个严格60 s达标时刻判定未通过")
    if radial_increment > 1e-7 or balance_relative > 0.002:
        raise RuntimeError("Q3单调性或水分守恒验收未通过")

    convergence.to_csv(validation_dir / "q3_grid_convergence.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(validation_dir / "q3_bdf_radau_comparison.csv", index=False, encoding="utf-8-sig")
    time_refinement.to_csv(validation_dir / "q3_time_refinement.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(validation_dir / "q3_sensitivity.csv", index=False, encoding="utf-8-sig")
    sensitivity_summary.to_csv(validation_dir / "q3_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(validation_dir / "q3_validation_summary.csv", index=False, encoding="utf-8-sig")
    return solution, convergence, comparison, time_refinement, sensitivity, summary


def cumulative_shrinking_moisture_balance(solution, boundary):
    time_s = solution["time_s"]
    moisture = solution["moisture"]
    _, _, material_volume, material_area = solution["unit_grid"]
    mean_moisture = moisture @ material_volume / material_volume.sum()
    _, air_moisture = boundary_values(
        time_s, boundary, long_term_mode=solution["long_term_mode"]
    )
    outward_rate = (
        material_area[-1]
        * H_M
        * solution["mass_transfer_scale"]
        * (moisture[:, -1] - air_moisture)
        / (solution["radius_at_time_m"] * material_volume.sum())
    )
    cumulative = np.zeros_like(time_s)
    cumulative[1:] = np.cumsum(
        0.5 * np.diff(time_s) * (outward_rate[:-1] + outward_rate[1:])
    )
    residual = mean_moisture - mean_moisture[0] + cumulative
    relative = np.max(np.abs(residual)) / max(
        abs(mean_moisture[0] - mean_moisture[-1]), 1e-12
    )
    return relative, residual[-1]


def run_q4_validation(
    boundary, radius_history, q3_solution=None, output_root=Path("outputs")
):
    validation_dir = output_root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    endpoints = {
        interval_count: solve_q4_endpoint(
            boundary,
            radius_history,
            interval_count=interval_count,
            max_step_s=300.0,
        )
        for interval_count in [80, 160, 320]
    }
    convergence = _endpoint_grid_frame(endpoints)
    finest_endpoint = endpoints[320]
    solution = solve_q4(
        boundary,
        radius_history,
        interval_count=320,
        endpoint=finest_endpoint,
    )

    radau = solve_q4_endpoint(
        boundary,
        radius_history,
        interval_count=80,
        method="Radau",
        max_step_s=300.0,
    )
    comparison = pd.DataFrame(
        [
            {
                "interval_count": 80,
                "primary_method": "BDF",
                "check_method": "Radau",
                "bdf_endpoint_s": endpoints[80]["endpoint_s"],
                "radau_endpoint_s": radau["endpoint_s"],
                "absolute_difference_s": abs(
                    endpoints[80]["endpoint_s"] - radau["endpoint_s"]
                ),
            }
        ]
    )
    tightened = solve_q4_endpoint(
        boundary,
        radius_history,
        interval_count=320,
        max_step_s=150.0,
        relative_tolerance=1e-8,
        temperature_tolerance=1e-8,
        moisture_tolerance=1e-10,
    )
    time_refinement = pd.DataFrame(
        [
            {
                "interval_count": 320,
                "baseline_endpoint_s": finest_endpoint["endpoint_s"],
                "tightened_endpoint_s": tightened["endpoint_s"],
                "absolute_difference_s": abs(
                    finest_endpoint["endpoint_s"] - tightened["endpoint_s"]
                ),
            }
        ]
    )

    shrinkage_rows = []
    for shrinkage_scale in [0.9, 1.0, 1.1]:
        endpoint = (
            endpoints[160]
            if shrinkage_scale == 1.0
            else solve_q4_endpoint(
                boundary,
                radius_history,
                interval_count=160,
                max_step_s=300.0,
                shrinkage_scale=shrinkage_scale,
            )
        )
        shrinkage_rows.append(
            {
                "scenario": "shrinkage_scale",
                "shrinkage_scale": shrinkage_scale,
                "radius_method": "pchip",
                "endpoint_s": endpoint["endpoint_s"],
                "endpoint_h": endpoint["endpoint_s"] / 3600.0,
                "endpoint_radius_cm": endpoint["endpoint_radius_m"] * 100.0,
            }
        )
    linear_endpoint = solve_q4_endpoint(
        boundary,
        radius_history,
        interval_count=160,
        max_step_s=300.0,
        radius_method="linear",
    )
    shrinkage_rows.append(
        {
            "scenario": "radius_interpolation",
            "shrinkage_scale": 1.0,
            "radius_method": "linear",
            "endpoint_s": linear_endpoint["endpoint_s"],
            "endpoint_h": linear_endpoint["endpoint_s"] / 3600.0,
            "endpoint_radius_cm": linear_endpoint["endpoint_radius_m"] * 100.0,
        }
    )
    sensitivity = pd.DataFrame(shrinkage_rows)
    sensitivity["delta_from_baseline_h"] = (
        sensitivity["endpoint_s"] - endpoints[160]["endpoint_s"]
    ) / 3600.0

    fixed_q4 = solve_q4_endpoint(
        boundary,
        radius_history,
        interval_count=320,
        max_step_s=300.0,
        fixed_radius_m=RADIUS_M,
    )
    if q3_solution is None:
        q3_endpoint_s = solve_q3_endpoint(
            boundary, interval_count=640, max_step_s=300.0
        )["endpoint_s"]
    else:
        q3_endpoint_s = q3_solution["endpoint_s"]
    decomposition = pd.DataFrame(
        [
            {"case": "P3_fixed_radius", "endpoint_s": q3_endpoint_s},
            {"case": "P4_fixed_radius", "endpoint_s": fixed_q4["endpoint_s"]},
            {"case": "P4_shrinking_radius", "endpoint_s": finest_endpoint["endpoint_s"]},
        ]
    )
    decomposition["endpoint_h"] = decomposition["endpoint_s"] / 3600.0
    decomposition["delta_from_P3_fixed_h"] = (
        decomposition["endpoint_s"] - q3_endpoint_s
    ) / 3600.0

    dense_time = np.arange(
        0.0, radius_history["time_s"][-1] + 1.0, 60.0
    )
    dense_radius = radius_values(dense_time, radius_history, method="pchip")
    radius_node_error = np.max(
        np.abs(
            radius_values(radius_history["time_s"], radius_history, method="pchip")
            - radius_history["radius_m"]
        )
    )
    radius_increment = np.max(np.diff(dense_radius))

    unit_grid = build_radial_grid(40, radius_m=1.0, surface_refinement=1.5)
    physical_grid = scale_material_grid(unit_grid, RADIUS_M)
    node_count = len(unit_grid[0])
    test_temperature = np.linspace(40.0, 48.0, node_count)
    test_moisture = np.linspace(1.2, 0.6, node_count)
    test_state = np.concatenate([test_temperature, test_moisture])
    dynamic_rate = q4_rhs(
        18000.0,
        test_state,
        boundary,
        unit_grid,
        radius_history,
        "pchip",
        1.0,
        RADIUS_M,
        1.0,
        1.0,
        "rounded",
    )
    air_temperature, air_moisture = boundary_values(18000.0, boundary)
    density, heat_capacity, conductivity, diffusivity = properties_q4(
        test_temperature, test_moisture
    )
    reference_rate = np.concatenate(
        [
            radial_rate(
                test_temperature,
                conductivity,
                density * heat_capacity,
                air_temperature,
                H_T,
                physical_grid,
            ),
            radial_rate(
                test_moisture,
                diffusivity,
                1.0,
                air_moisture,
                H_M,
                physical_grid,
            ),
        ]
    )
    fixed_domain_rate_error = np.max(np.abs(dynamic_rate - reference_rate))

    artificial_boundary = {
        "time_s": np.array([0.0, 60.0]),
        "air_temperature_c": np.array([28.0, 28.0]),
        "air_moisture": np.array([INITIAL_MOISTURE, INITIAL_MOISTURE]),
    }
    uniform_state = np.concatenate(
        [np.full(node_count, 28.0), np.full(node_count, INITIAL_MOISTURE)]
    )
    uniform_rate = q4_rhs(
        0.0,
        uniform_state,
        artificial_boundary,
        unit_grid,
        radius_history,
        "pchip",
        1.0,
        RADIUS_M,
        1.0,
        1.0,
        "rounded",
    )

    balance_relative, balance_final = cumulative_shrinking_moisture_balance(
        solution, boundary
    )
    max_before = np.max(solution["moisture"][-2])
    max_strict = np.max(solution["moisture"][-1])
    endpoint_index = int(np.argmax(solution["endpoint_moisture"]))
    radial_increment = np.max(np.diff(solution["moisture"], axis=1))
    density, heat_capacity, conductivity, diffusivity = properties_q4(
        solution["temperature_c"], solution["moisture"]
    )
    summary = pd.DataFrame(
        [
            {"check": "continuous_endpoint", "value": solution["endpoint_s"], "unit": "s"},
            {"check": "continuous_endpoint", "value": solution["endpoint_s"] / 3600.0, "unit": "h"},
            {"check": "strict_60s_endpoint", "value": solution["strict_output_s"], "unit": "s"},
            {"check": "endpoint_radius", "value": solution["endpoint_radius_m"] * 100.0, "unit": "cm"},
            {"check": "maximum_moisture_before_strict", "value": max_before, "unit": "kg/kg"},
            {"check": "maximum_moisture_at_strict", "value": max_strict, "unit": "kg/kg"},
            {"check": "endpoint_control_material_coordinate", "value": solution["material_coordinate"][endpoint_index], "unit": "1"},
            {"check": "maximum_radial_moisture_increment", "value": radial_increment, "unit": "kg/kg"},
            {"check": "cumulative_moisture_balance_relative", "value": balance_relative, "unit": "1"},
            {"check": "cumulative_moisture_balance_final", "value": balance_final, "unit": "kg/kg"},
            {"check": "radius_node_interpolation_error", "value": radius_node_error, "unit": "m"},
            {"check": "maximum_radius_increment", "value": radius_increment, "unit": "m"},
            {"check": "fixed_radius_rate_max_difference", "value": fixed_domain_rate_error, "unit": "state/s"},
            {"check": "uniform_field_rhs_max", "value": np.max(np.abs(uniform_rate)), "unit": "state/s"},
            {"check": "minimum_density", "value": density.min(), "unit": "kg/m3"},
            {"check": "minimum_heat_capacity", "value": heat_capacity.min(), "unit": "J/(kg K)"},
            {"check": "minimum_conductivity", "value": conductivity.min(), "unit": "W/(m K)"},
            {"check": "minimum_diffusivity", "value": diffusivity.min(), "unit": "m2/s"},
        ]
    )

    finest_grid_difference = abs(
        endpoints[320]["endpoint_s"] - endpoints[160]["endpoint_s"]
    )
    if finest_grid_difference > 30.0:
        raise RuntimeError(
            f"Q4终点网格验收未通过：160到320区间差{finest_grid_difference:.3f} s"
        )
    if time_refinement["absolute_difference_s"].iloc[0] > 5.0:
        raise RuntimeError("Q4时间积分收紧验收未通过")
    if comparison["absolute_difference_s"].iloc[0] > 5.0:
        raise RuntimeError("Q4 BDF与Radau终点复核未通过")
    if max_before < 0.15 or max_strict >= 0.15:
        raise RuntimeError("Q4首个严格60 s达标时刻判定未通过")
    if radial_increment > 1e-7 or balance_relative > 0.002:
        raise RuntimeError("Q4单调性或水分守恒验收未通过")
    if radius_node_error > 1e-12 or radius_increment > 1e-12:
        raise RuntimeError("Q4半径PCHIP插值验收未通过")
    if fixed_domain_rate_error > 1e-12 or np.max(np.abs(uniform_rate)) > 1e-12:
        raise RuntimeError("Q4固定半径退化或均匀场验收未通过")

    convergence.to_csv(validation_dir / "q4_grid_convergence.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(validation_dir / "q4_bdf_radau_comparison.csv", index=False, encoding="utf-8-sig")
    time_refinement.to_csv(validation_dir / "q4_time_refinement.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(validation_dir / "q4_shrinkage_sensitivity.csv", index=False, encoding="utf-8-sig")
    decomposition.to_csv(validation_dir / "q4_effect_decomposition.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(validation_dir / "q4_validation_summary.csv", index=False, encoding="utf-8-sig")
    return (
        solution,
        convergence,
        comparison,
        time_refinement,
        sensitivity,
        decomposition,
        summary,
    )
