import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from data_processing import boundary_values, radius_values


RADIUS_M = 0.02
LENGTH_M = 0.25
INITIAL_TEMPERATURE_C = 28.0
INITIAL_MOISTURE = 2.55
RHO_Q1 = 820.0
CP_Q1 = 2600.0
K_Q1 = 0.36
H_T = 25.0
H_M = 8e-7
DRYING_THRESHOLD = 0.15


def build_radial_grid(interval_count, radius_m=RADIUS_M, surface_refinement=1.0):
    if interval_count < 2 or radius_m <= 0 or surface_refinement < 1.0:
        raise ValueError("网格区间数、半径或表面加密参数不合理")
    if surface_refinement == 1.0:
        radius = np.linspace(0.0, radius_m, interval_count + 1)
    else:
        uniform_coordinate = np.linspace(0.0, 1.0, interval_count + 1)
        radius = radius_m * (
            1.0 - (1.0 - uniform_coordinate) ** surface_refinement
        )
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
    spacing = np.diff(radius)
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


def q3_rhs(
    time_s,
    state,
    boundary,
    grid,
    diffusivity_scale,
    mass_transfer_scale,
    long_term_mode,
):
    node_count = len(grid[0])
    temperature_c = state[:node_count]
    moisture = state[node_count:]
    air_temperature_c, air_moisture = boundary_values(
        time_s, boundary, long_term_mode=long_term_mode
    )
    density, heat_capacity, conductivity, diffusivity = properties_q23(
        temperature_c, moisture
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
        diffusivity_scale * diffusivity,
        1.0,
        air_moisture,
        mass_transfer_scale * H_M,
        grid,
    )
    return np.concatenate([temperature_rate, moisture_rate])


def _initial_state(node_count):
    return np.concatenate(
        [
            np.full(node_count, INITIAL_TEMPERATURE_C),
            np.full(node_count, INITIAL_MOISTURE),
        ]
    )


def _absolute_tolerance(
    node_count, temperature_tolerance=1e-7, moisture_tolerance=1e-9
):
    return np.concatenate(
        [
            np.full(node_count, temperature_tolerance),
            np.full(node_count, moisture_tolerance),
        ]
    )


def _drying_event(node_count, threshold):
    def event(_time_s, state, *_args):
        return np.max(state[node_count:]) - threshold

    event.terminal = True
    event.direction = -1
    return event


def _endpoint_from_result(result, node_count, label):
    if not result.success:
        raise RuntimeError(f"{label}求解失败：{result.message}")
    if not len(result.t_events[0]):
        raise RuntimeError(f"{label}在最大计算时长内未达到全域含水率阈值")
    endpoint_s = float(result.t_events[0][0])
    endpoint_state = result.y_events[0][0]
    return {
        "endpoint_s": endpoint_s,
        "endpoint_temperature_c": endpoint_state[:node_count],
        "endpoint_moisture": endpoint_state[node_count:],
        "solver_steps": result.nfev,
        "solver_message": result.message,
    }


def solve_q3_endpoint(
    boundary,
    interval_count=160,
    method="BDF",
    max_time_s=14 * 86400.0,
    max_step_s=300.0,
    relative_tolerance=1e-7,
    temperature_tolerance=1e-7,
    moisture_tolerance=1e-9,
    surface_refinement=1.5,
    diffusivity_scale=1.0,
    mass_transfer_scale=1.0,
    long_term_mode="rounded",
    threshold=DRYING_THRESHOLD,
):
    grid = build_radial_grid(
        interval_count, surface_refinement=surface_refinement
    )
    node_count = len(grid[0])
    result = solve_ivp(
        q3_rhs,
        (0.0, max_time_s),
        _initial_state(node_count),
        args=(
            boundary,
            grid,
            diffusivity_scale,
            mass_transfer_scale,
            long_term_mode,
        ),
        method=method,
        rtol=relative_tolerance,
        atol=_absolute_tolerance(
            node_count, temperature_tolerance, moisture_tolerance
        ),
        max_step=max_step_s,
        jac_sparsity=q2_jacobian_sparsity(node_count),
        events=_drying_event(node_count, threshold),
    )
    endpoint = _endpoint_from_result(result, node_count, f"Q3 {method}")
    endpoint.update(
        {
            "radius_m": grid[0],
            "grid": grid,
            "method": method,
            "max_step_s": max_step_s,
            "diffusivity_scale": diffusivity_scale,
            "mass_transfer_scale": mass_transfer_scale,
            "long_term_mode": long_term_mode,
            "relative_tolerance": relative_tolerance,
            "temperature_tolerance": temperature_tolerance,
            "moisture_tolerance": moisture_tolerance,
            "surface_refinement": surface_refinement,
        }
    )
    return endpoint


def solve_q3(boundary, interval_count=640, endpoint=None, **endpoint_options):
    if endpoint is None:
        endpoint = solve_q3_endpoint(
            boundary, interval_count=interval_count, **endpoint_options
        )
    strict_output_s = 60.0 * (np.floor(endpoint["endpoint_s"] / 60.0) + 1.0)
    output_time_s = np.arange(0.0, strict_output_s + 1.0, 60.0)
    grid = build_radial_grid(
        interval_count, surface_refinement=endpoint["surface_refinement"]
    )
    node_count = len(grid[0])
    result = solve_ivp(
        q3_rhs,
        (0.0, strict_output_s),
        _initial_state(node_count),
        args=(
            boundary,
            grid,
            endpoint["diffusivity_scale"],
            endpoint["mass_transfer_scale"],
            endpoint["long_term_mode"],
        ),
        method=endpoint["method"],
        t_eval=output_time_s,
        rtol=endpoint["relative_tolerance"],
        atol=_absolute_tolerance(
            node_count,
            endpoint["temperature_tolerance"],
            endpoint["moisture_tolerance"],
        ),
        max_step=endpoint["max_step_s"],
        jac_sparsity=q2_jacobian_sparsity(node_count),
    )
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError(f"Q3正式输出求解失败：{result.message}")
    temperature_c = result.y[:node_count].T
    moisture = result.y[node_count:].T
    if np.max(moisture[-1]) >= DRYING_THRESHOLD:
        raise RuntimeError("Q3首个60 s输出时刻未严格满足全域含水率要求")
    return {
        "time_s": result.t,
        "radius_m": grid[0],
        "temperature_c": temperature_c,
        "moisture": moisture,
        "grid": grid,
        "strict_output_s": strict_output_s,
        **endpoint,
    }


def properties_q4(temperature_c, moisture):
    temperature_k = temperature_c + 273.15
    if np.any(temperature_k < 250.0) or np.any(temperature_k > 400.0):
        raise ValueError("Q4温度超出经验公式的合理K值范围")
    if np.any(moisture <= 0):
        raise ValueError("Q4含水率出现非正值")
    density = 760.0 + 90.0 * moisture
    heat_capacity = 1850.0 + 2150.0 * moisture / (moisture + 1.0)
    conductivity = 0.12 + 0.20 * moisture / (moisture + 1.0)
    diffusivity = (
        4.2e-4
        * np.exp(-0.30 / moisture)
        * np.exp(-3850.0 / temperature_k)
    )
    if np.any(density <= 0) or np.any(heat_capacity <= 0):
        raise ValueError("Q4密度或比热容非正")
    if np.any(conductivity <= 0) or np.any(diffusivity <= 0):
        raise ValueError("Q4导热系数或扩散系数非正")
    return density, heat_capacity, conductivity, diffusivity


def scale_material_grid(unit_grid, radius_m):
    coordinate, faces, unit_volume, unit_area = unit_grid
    return (
        coordinate * radius_m,
        faces * radius_m,
        unit_volume * radius_m**2,
        unit_area * radius_m,
    )


def q4_rhs(
    time_s,
    state,
    boundary,
    unit_grid,
    radius_history,
    radius_method,
    shrinkage_scale,
    fixed_radius_m,
    diffusivity_scale,
    mass_transfer_scale,
    long_term_mode,
):
    node_count = len(unit_grid[0])
    temperature_c = state[:node_count]
    moisture = state[node_count:]
    current_radius_m = (
        fixed_radius_m
        if fixed_radius_m is not None
        else radius_values(
            time_s,
            radius_history,
            method=radius_method,
            shrinkage_scale=shrinkage_scale,
        )
    )
    physical_grid = scale_material_grid(unit_grid, current_radius_m)
    air_temperature_c, air_moisture = boundary_values(
        time_s, boundary, long_term_mode=long_term_mode
    )
    density, heat_capacity, conductivity, diffusivity = properties_q4(
        temperature_c, moisture
    )
    temperature_rate = radial_rate(
        temperature_c,
        conductivity,
        density * heat_capacity,
        air_temperature_c,
        H_T,
        physical_grid,
    )
    moisture_rate = radial_rate(
        moisture,
        diffusivity_scale * diffusivity,
        1.0,
        air_moisture,
        mass_transfer_scale * H_M,
        physical_grid,
    )
    return np.concatenate([temperature_rate, moisture_rate])


def solve_q4_endpoint(
    boundary,
    radius_history,
    interval_count=160,
    method="BDF",
    max_time_s=7 * 86400.0,
    max_step_s=300.0,
    relative_tolerance=1e-7,
    temperature_tolerance=1e-7,
    moisture_tolerance=1e-9,
    surface_refinement=1.5,
    radius_method="pchip",
    shrinkage_scale=1.0,
    fixed_radius_m=None,
    diffusivity_scale=1.0,
    mass_transfer_scale=1.0,
    long_term_mode="rounded",
    threshold=DRYING_THRESHOLD,
):
    unit_grid = build_radial_grid(
        interval_count, radius_m=1.0, surface_refinement=surface_refinement
    )
    node_count = len(unit_grid[0])
    args = (
        boundary,
        unit_grid,
        radius_history,
        radius_method,
        shrinkage_scale,
        fixed_radius_m,
        diffusivity_scale,
        mass_transfer_scale,
        long_term_mode,
    )
    result = solve_ivp(
        q4_rhs,
        (0.0, max_time_s),
        _initial_state(node_count),
        args=args,
        method=method,
        rtol=relative_tolerance,
        atol=_absolute_tolerance(
            node_count, temperature_tolerance, moisture_tolerance
        ),
        max_step=max_step_s,
        jac_sparsity=q2_jacobian_sparsity(node_count),
        events=_drying_event(node_count, threshold),
    )
    endpoint = _endpoint_from_result(result, node_count, f"Q4 {method}")
    endpoint_radius_m = (
        fixed_radius_m
        if fixed_radius_m is not None
        else radius_values(
            endpoint["endpoint_s"],
            radius_history,
            method=radius_method,
            shrinkage_scale=shrinkage_scale,
        )
    )
    endpoint.update(
        {
            "material_coordinate": unit_grid[0],
            "unit_grid": unit_grid,
            "endpoint_radius_m": endpoint_radius_m,
            "method": method,
            "max_step_s": max_step_s,
            "radius_method": radius_method,
            "shrinkage_scale": shrinkage_scale,
            "fixed_radius_m": fixed_radius_m,
            "diffusivity_scale": diffusivity_scale,
            "mass_transfer_scale": mass_transfer_scale,
            "long_term_mode": long_term_mode,
            "relative_tolerance": relative_tolerance,
            "temperature_tolerance": temperature_tolerance,
            "moisture_tolerance": moisture_tolerance,
            "surface_refinement": surface_refinement,
        }
    )
    return endpoint


def solve_q4(
    boundary,
    radius_history,
    interval_count=320,
    endpoint=None,
    **endpoint_options,
):
    if endpoint is None:
        endpoint = solve_q4_endpoint(
            boundary,
            radius_history,
            interval_count=interval_count,
            **endpoint_options,
        )
    strict_output_s = 60.0 * (np.floor(endpoint["endpoint_s"] / 60.0) + 1.0)
    output_time_s = np.arange(0.0, strict_output_s + 1.0, 60.0)
    unit_grid = build_radial_grid(
        interval_count,
        radius_m=1.0,
        surface_refinement=endpoint["surface_refinement"],
    )
    node_count = len(unit_grid[0])
    args = (
        boundary,
        unit_grid,
        radius_history,
        endpoint["radius_method"],
        endpoint["shrinkage_scale"],
        endpoint["fixed_radius_m"],
        endpoint["diffusivity_scale"],
        endpoint["mass_transfer_scale"],
        endpoint["long_term_mode"],
    )
    result = solve_ivp(
        q4_rhs,
        (0.0, strict_output_s),
        _initial_state(node_count),
        args=args,
        method=endpoint["method"],
        t_eval=output_time_s,
        rtol=endpoint["relative_tolerance"],
        atol=_absolute_tolerance(
            node_count,
            endpoint["temperature_tolerance"],
            endpoint["moisture_tolerance"],
        ),
        max_step=endpoint["max_step_s"],
        jac_sparsity=q2_jacobian_sparsity(node_count),
    )
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError(f"Q4正式输出求解失败：{result.message}")
    temperature_c = result.y[:node_count].T
    moisture = result.y[node_count:].T
    if np.max(moisture[-1]) >= DRYING_THRESHOLD:
        raise RuntimeError("Q4首个60 s输出时刻未严格满足全域含水率要求")
    radius_at_time_m = (
        np.full_like(result.t, endpoint["fixed_radius_m"], dtype=float)
        if endpoint["fixed_radius_m"] is not None
        else radius_values(
            result.t,
            radius_history,
            method=endpoint["radius_method"],
            shrinkage_scale=endpoint["shrinkage_scale"],
        )
    )
    return {
        "time_s": result.t,
        "material_coordinate": unit_grid[0],
        "temperature_c": temperature_c,
        "moisture": moisture,
        "radius_at_time_m": radius_at_time_m,
        "strict_output_s": strict_output_s,
        **endpoint,
    }
