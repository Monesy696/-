import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from data_processing import boundary_values


RADIUS_M = 0.02
LENGTH_M = 0.25
INITIAL_TEMPERATURE_C = 28.0
INITIAL_MOISTURE = 2.55
RHO_Q1 = 820.0
CP_Q1 = 2600.0
K_Q1 = 0.36
H_T = 25.0
H_M = 8e-7


def build_radial_grid(interval_count, radius_m=RADIUS_M):
    radius = np.linspace(0.0, radius_m, interval_count + 1)
    faces = np.empty(interval_count + 2)
    faces[0] = 0.0
    faces[-1] = radius_m
    faces[1:-1] = 0.5 * (radius[:-1] + radius[1:])

    volume = np.pi * LENGTH_M * (faces[1:] ** 2 - faces[:-1] ** 2)
    face_area = 2.0 * np.pi * LENGTH_M * faces
    return radius, faces, volume, face_area


def diffusivity_q1(moisture):
    if np.any(moisture <= 0):
        raise ValueError("Q1含水率出现非正值，请检查数值设置或边界符号")
    return 7e-9 * np.exp(-0.89 / moisture)


def harmonic_mean(left, right):
    return 2.0 * left * right / (left + right)


def radial_rate(state, coefficient, capacity, external_value, transfer_coefficient, grid):
    radius, _, volume, face_area = grid
    interval_count = len(radius) - 1
    spacing = radius[1] - radius[0]
    interface_coefficient = harmonic_mean(coefficient[:-1], coefficient[1:])

    flux = np.zeros(interval_count + 2)
    flux[1:-1] = -interface_coefficient * np.diff(state) / spacing
    flux[-1] = transfer_coefficient * (state[-1] - external_value)

    balance = face_area[:-1] * flux[:-1] - face_area[1:] * flux[1:]
    return balance / (capacity * volume)


def q1_rhs(time_s, state, boundary, grid):
    node_count = len(grid[0])
    temperature_c = state[:node_count]
    moisture = state[node_count:]
    air_temperature_c, air_moisture = boundary_values(time_s, boundary)

    temperature_rate = radial_rate(
        temperature_c,
        np.full(node_count, K_Q1),
        RHO_Q1 * CP_Q1,
        air_temperature_c,
        H_T,
        grid,
    )
    moisture_rate = radial_rate(
        moisture,
        diffusivity_q1(moisture),
        1.0,
        air_moisture,
        H_M,
        grid,
    )
    return np.concatenate([temperature_rate, moisture_rate])


def q1_jacobian_sparsity(node_count):
    size = 2 * node_count
    pattern = lil_matrix((size, size), dtype=int)
    for offset in [0, node_count]:
        for index in range(node_count):
            pattern[offset + index, offset + index] = 1
            if index > 0:
                pattern[offset + index, offset + index - 1] = 1
            if index < node_count - 1:
                pattern[offset + index, offset + index + 1] = 1
    return pattern.tocsr()


def solve_q1(boundary, interval_count=80, output_time_s=None):
    if output_time_s is None:
        output_time_s = np.arange(0.0, 1801.0, 1.0)
    output_time_s = np.asarray(output_time_s, dtype=float)
    if output_time_s[0] != 0 or output_time_s[-1] != 1800:
        raise ValueError("Q1输出时间必须覆盖0–1800 s")

    grid = build_radial_grid(interval_count)
    node_count = len(grid[0])
    initial_state = np.concatenate(
        [
            np.full(node_count, INITIAL_TEMPERATURE_C),
            np.full(node_count, INITIAL_MOISTURE),
        ]
    )
    absolute_tolerance = np.concatenate(
        [np.full(node_count, 1e-7), np.full(node_count, 1e-9)]
    )

    result = solve_ivp(
        q1_rhs,
        (0.0, 1800.0),
        initial_state,
        args=(boundary, grid),
        method="BDF",
        t_eval=output_time_s,
        rtol=1e-7,
        atol=absolute_tolerance,
        max_step=30.0,
        jac_sparsity=q1_jacobian_sparsity(node_count),
    )
    if not result.success:
        raise RuntimeError(f"Q1 BDF求解失败：{result.message}")

    temperature_c = result.y[:node_count].T
    moisture = result.y[node_count:].T
    if not np.isfinite(result.y).all():
        raise ValueError("Q1结果包含NaN或无穷值")
    if moisture.min() <= 0:
        raise ValueError("Q1结果出现非正含水率")

    return {
        "time_s": result.t,
        "radius_m": grid[0],
        "temperature_c": temperature_c,
        "moisture": moisture,
        "grid": grid,
        "solver_steps": result.nfev,
        "solver_message": result.message,
    }


def properties_q23(temperature_c, moisture):
    temperature_k = temperature_c + 273.15
    if np.any(temperature_k < 250.0) or np.any(temperature_k > 400.0):
        raise ValueError("Q2温度超出经验公式的合理K值范围")
    if np.any(moisture <= 0):
        raise ValueError("Q2含水率出现非正值")

    density = 650.0 + 128.0 * moisture
    heat_capacity = 1450.0 + 2736.0 * moisture / (moisture + 1.0)
    conductivity = 0.21 + 0.38 * moisture / (moisture + 1.0)
    diffusivity = (
        2.4e-3
        * np.exp(-0.45 / moisture)
        * np.exp(-3850.0 / temperature_k)
    )
    if np.any(density <= 0) or np.any(heat_capacity <= 0):
        raise ValueError("Q2密度或比热容非正")
    if np.any(conductivity <= 0) or np.any(diffusivity <= 0):
        raise ValueError("Q2导热系数或扩散系数非正")
    return density, heat_capacity, conductivity, diffusivity


def q2_rhs(time_s, state, boundary, grid):
    node_count = len(grid[0])
    temperature_c = state[:node_count]
    moisture = state[node_count:]
    air_temperature_c, air_moisture = boundary_values(time_s, boundary)
    density, heat_capacity, conductivity, diffusivity = properties_q23(
        temperature_c,
        moisture,
    )

    temperature_rate = radial_rate(
        temperature_c,
        conductivity,
        density * heat_capacity,
        air_temperature_c,
        H_T,
        grid,
    )
    moisture_rate = radial_rate(
        moisture,
        diffusivity,
        1.0,
        air_moisture,
        H_M,
        grid,
    )
    return np.concatenate([temperature_rate, moisture_rate])


def q2_jacobian_sparsity(node_count):
    size = 2 * node_count
    pattern = lil_matrix((size, size), dtype=int)
    for equation_offset in [0, node_count]:
        for variable_offset in [0, node_count]:
            for index in range(node_count):
                for neighbor in [index - 1, index, index + 1]:
                    if 0 <= neighbor < node_count:
                        pattern[equation_offset + index, variable_offset + neighbor] = 1
    return pattern.tocsr()


def solve_q2(boundary, interval_count=160, output_time_s=None, method="BDF"):
    if output_time_s is None:
        output_time_s = np.arange(0.0, 10801.0, 1.0)
    output_time_s = np.asarray(output_time_s, dtype=float)
    if output_time_s[0] != 0 or output_time_s[-1] != 10800:
        raise ValueError("Q2输出时间必须覆盖0–10800 s")
    if np.any(np.diff(output_time_s) <= 0):
        raise ValueError("Q2输出时间必须严格递增")

    grid = build_radial_grid(interval_count)
    node_count = len(grid[0])
    initial_state = np.concatenate(
        [
            np.full(node_count, INITIAL_TEMPERATURE_C),
            np.full(node_count, INITIAL_MOISTURE),
        ]
    )
    absolute_tolerance = np.concatenate(
        [np.full(node_count, 1e-7), np.full(node_count, 1e-9)]
    )
    result = solve_ivp(
        q2_rhs,
        (0.0, 10800.0),
        initial_state,
        args=(boundary, grid),
        method=method,
        t_eval=output_time_s,
        rtol=1e-7,
        atol=absolute_tolerance,
        max_step=30.0,
        jac_sparsity=q2_jacobian_sparsity(node_count),
    )
    if not result.success:
        raise RuntimeError(f"Q2 {method}求解失败：{result.message}")
    if not np.isfinite(result.y).all():
        raise ValueError("Q2结果包含NaN或无穷值")

    temperature_c = result.y[:node_count].T
    moisture = result.y[node_count:].T
    if moisture.min() <= 0:
        raise ValueError("Q2结果出现非正含水率")
    return {
        "time_s": result.t,
        "radius_m": grid[0],
        "temperature_c": temperature_c,
        "moisture": moisture,
        "grid": grid,
        "solver_steps": result.nfev,
        "solver_message": result.message,
        "method": method,
    }
