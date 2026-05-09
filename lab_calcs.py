#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Moto: laboratory test calculations

Calculations for equivalent-circuit estimation from no-load and locked-rotor
laboratory tests. The existing optimisation-based estimation flow is kept
separate and untouched.
"""

import math
import numpy as np


COPPER_TEMP_CONSTANT = 234.5
EPSILON = 1e-9


def synchronous_speed(frequency_hz, pole_count):
    return 120.0 * frequency_hz / pole_count


def rated_torque(power_w, speed_rpm):
    omega_mech = 2.0 * math.pi * speed_rpm / 60.0
    if abs(omega_mech) < EPSILON:
        raise ValueError('Rated speed must be greater than zero.')
    return power_w / omega_mech


def correct_resistance(resistance_ohm, source_temp_c, target_temp_c):
    return resistance_ohm * ((target_temp_c + COPPER_TEMP_CONSTANT) / (source_temp_c + COPPER_TEMP_CONSTANT))


def equivalent_phase_resistance(lab_data):
    resistance_ohm = lab_data['blocked_resistance_ohm']
    input_mode = str(lab_data.get('resistance_input_mode', 'phase_equivalent')).lower()
    connection = str(lab_data.get('stator_connection', 'WYE')).upper()

    if input_mode == 'line_to_line':
        return resistance_ohm / 2.0
    if input_mode == 'phase_winding':
        if connection == 'DELTA':
            return resistance_ohm / 3.0
        return resistance_ohm
    return resistance_ohm


def estimate_single_cage_parameters(lab_data):
    rated_frequency = lab_data['frequency_hz']
    blocked_frequency = lab_data['blocked_frequency_hz']
    blocked_voltage_phase = lab_data['blocked_voltage_v'] / math.sqrt(3.0)
    blocked_current = lab_data['blocked_current_a']
    reactance_ratio = float(lab_data.get('reactance_ratio', 1.0))
    include_core_loss = bool(lab_data.get('include_core_loss', True))

    if blocked_current <= 0.0:
        raise ValueError('Blocked-rotor current must be greater than zero.')
    if blocked_frequency <= 0.0:
        raise ValueError('Blocked-rotor frequency must be greater than zero.')
    if reactance_ratio <= 0.0:
        raise ValueError('Reactance ratio must be greater than zero.')

    rs_reference = equivalent_phase_resistance(lab_data)
    resistance_measurement_temp = lab_data.get('resistance_measurement_temp_c', lab_data['blocked_temp_c'])
    rs = correct_resistance(rs_reference, resistance_measurement_temp, lab_data['blocked_temp_c'])
    z_blocked = blocked_voltage_phase / blocked_current
    r_total = lab_data['blocked_power_w'] / (3.0 * blocked_current ** 2)
    rr1 = max(r_total - rs, EPSILON)

    x_total_test = math.sqrt(max(z_blocked ** 2 - r_total ** 2, EPSILON))
    x_total_nominal = x_total_test * rated_frequency / blocked_frequency
    xs = x_total_nominal / (1.0 + reactance_ratio)
    xr1 = reactance_ratio * xs

    no_load_voltage_phase = lab_data['no_load_voltage_v'] / math.sqrt(3.0)
    no_load_current = lab_data['no_load_current_a']
    if no_load_current <= 0.0:
        raise ValueError('No-load current must be greater than zero.')
    rs_no_load = correct_resistance(rs_reference, resistance_measurement_temp, lab_data['no_load_temp_c'])
    no_load_copper_loss = 3.0 * no_load_current ** 2 * rs_no_load
    rotational_loss = max(lab_data['no_load_power_w'] - no_load_copper_loss, 0.0)

    if include_core_loss:
        core_loss = max(lab_data['no_load_power_w'] - lab_data['fw_loss_w'] - no_load_copper_loss, EPSILON)
        rc = 3.0 * no_load_voltage_phase ** 2 / core_loss
        in_phase_current = no_load_voltage_phase / rc
        magnetizing_current = math.sqrt(max(no_load_current ** 2 - in_phase_current ** 2, EPSILON))
        xm = no_load_voltage_phase / magnetizing_current
    else:
        x_total_no_load = math.sqrt(max((no_load_voltage_phase / no_load_current) ** 2 - rs_no_load ** 2, EPSILON))
        xm = max(x_total_no_load - xs, EPSILON)
        rc = None

    return {
        'Rs': rs,
        'Xs': xs,
        'Xm': xm,
        'Rr1': rr1,
        'Xr1': xr1,
        'Rc': rc,
        'reactance_ratio': reactance_ratio,
        'rotational_loss_w': rotational_loss,
        'include_core_loss': include_core_loss,
    }


def single_cage_performance(lab_data, params, slip, line_voltage_v=None):
    slip = min(max(slip, 1e-4), 1.0)
    line_voltage = lab_data['rated_voltage_v'] if line_voltage_v is None else line_voltage_v
    phase_voltage = line_voltage / math.sqrt(3.0)
    sync_speed_rpm = synchronous_speed(lab_data['frequency_hz'], lab_data['pole_count'])
    omega_sync = 2.0 * math.pi * sync_speed_rpm / 60.0

    zs = complex(params['Rs'], params['Xs'])
    if params.get('Rc') is None:
        ym = complex(0.0, -1.0 / params['Xm'])
        shaft_loss = params.get('rotational_loss_w', lab_data['fw_loss_w'])
    else:
        ym = complex(1.0 / params['Rc'], -1.0 / params['Xm'])
        shaft_loss = lab_data['fw_loss_w']
    zr = complex(params['Rr1'] / slip, params['Xr1'])

    y_parallel = ym + (1.0 / zr)
    z_parallel = 1.0 / y_parallel
    z_total = zs + z_parallel

    stator_current = phase_voltage / z_total
    air_gap_voltage = phase_voltage - stator_current * zs
    rotor_current = air_gap_voltage / zr

    apparent_power = 3.0 * phase_voltage * np.conj(stator_current)
    input_power = float(np.real(apparent_power))
    reactive_power = abs(float(np.imag(apparent_power)))

    air_gap_power = 3.0 * (abs(rotor_current) ** 2) * params['Rr1'] / slip
    converted_power = air_gap_power * (1.0 - slip)
    shaft_power = max(converted_power - shaft_loss, 0.0)
    torque_nm = air_gap_power / omega_sync

    current_a = abs(stator_current)
    apparent_magnitude = math.sqrt(3.0) * line_voltage * current_a
    power_factor = input_power / apparent_magnitude if apparent_magnitude > EPSILON else 0.0
    power_factor = min(max(power_factor, 0.0), 1.0)

    efficiency = shaft_power / input_power if input_power > EPSILON else 0.0
    efficiency = min(max(efficiency, 0.0), 1.0)

    return {
        'slip': slip,
        'speed_rpm': sync_speed_rpm * (1.0 - slip),
        'torque_nm': torque_nm,
        'current_a': current_a,
        'input_power_w': input_power,
        'reactive_power_var': reactive_power,
        'shaft_power_w': shaft_power,
        'power_factor': power_factor,
        'efficiency': efficiency,
    }


def single_cage_no_load_slip(lab_data, params, line_voltage_v=None):
    slips = np.geomspace(1e-4, 0.2, 600)
    best_slip = float(slips[0])
    best_error = None

    for slip in slips:
        point = single_cage_performance(lab_data, params, float(slip), line_voltage_v=line_voltage_v)
        error = abs(point['shaft_power_w'])
        if best_error is None or error < best_error:
            best_slip = float(slip)
            best_error = error

    return best_slip


def single_cage_lab_metrics(lab_data, params):
    no_load_slip = single_cage_no_load_slip(
        lab_data,
        params,
        line_voltage_v=lab_data['no_load_voltage_v'],
    )
    no_load = single_cage_performance(
        lab_data,
        params,
        no_load_slip,
        line_voltage_v=lab_data['no_load_voltage_v'],
    )
    blocked = single_cage_performance(
        lab_data,
        params,
        1.0,
        line_voltage_v=lab_data['blocked_voltage_v'],
    )
    starting = single_cage_performance(
        lab_data,
        params,
        1.0,
        line_voltage_v=lab_data['rated_voltage_v'],
    )
    rated_slip = (synchronous_speed(lab_data['frequency_hz'], lab_data['pole_count']) - lab_data['rated_speed_rpm'])
    rated_slip /= synchronous_speed(lab_data['frequency_hz'], lab_data['pole_count'])
    rated = single_cage_performance(lab_data, params, rated_slip)

    return {
        'no_load_current_a': no_load['current_a'],
        'no_load_power_w': no_load['input_power_w'],
        'blocked_current_a': blocked['current_a'],
        'blocked_power_w': blocked['input_power_w'],
        'starting_current_a': starting['current_a'],
        'rated_current_a': rated['current_a'],
        'rated_pf': rated['power_factor'],
        'rated_eff': rated['efficiency'],
        'rated_power_w': rated['shaft_power_w'],
    }


def single_cage_fit_error(lab_data, params):
    metrics = single_cage_lab_metrics(lab_data, params)
    weights = {
        'no_load_current_a': 1.0,
        'no_load_power_w': 1.0,
        'blocked_current_a': 0.5 if lab_data.get('starting_current_target_a', 0.0) > 0.0 else 1.0,
        'blocked_power_w': 0.5 if lab_data.get('starting_current_target_a', 0.0) > 0.0 else 1.0,
        'starting_current_a': 1.0,
        'rated_current_a': 1.25,
        'rated_pf': 1.0,
        'rated_eff': 1.0,
        'rated_power_w': 1.25,
    }
    targets = {
        'no_load_current_a': lab_data.get('no_load_current_target_a', lab_data['no_load_current_a']),
        'no_load_power_w': lab_data.get('no_load_power_target_w', lab_data['no_load_power_w']),
        'blocked_current_a': lab_data['blocked_current_a'],
        'blocked_power_w': lab_data['blocked_power_w'],
        'starting_current_a': lab_data.get('starting_current_target_a', 0.0),
        'rated_current_a': lab_data.get('rated_current_target_a', lab_data['rated_current_a']),
        'rated_pf': lab_data.get('rated_pf_target', 0.0),
        'rated_eff': lab_data.get('rated_eff_target', 0.0),
        'rated_power_w': lab_data.get('rated_power_target_w', lab_data['rated_power_w']),
    }

    error = 0.0
    for key, target in targets.items():
        if target is None or target <= 0.0:
            continue
        delta = (metrics[key] - target) / max(abs(target), EPSILON)
        error += weights[key] * delta * delta

    return error


def refine_single_cage_parameters(lab_data, initial_params, max_iter=80):
    variable_keys = ['Xs', 'Xm', 'Rr1']
    if initial_params.get('Rc') is not None:
        variable_keys.append('Rc')
    if lab_data.get('fit_reactance_ratio', False):
        variable_keys.append('reactance_ratio')

    best_values = np.array([max(float(initial_params[key]), EPSILON) for key in variable_keys], dtype=float)
    step = math.log(1.35)

    def build_params(values):
        params = dict(initial_params)
        for index, key in enumerate(variable_keys):
            params[key] = max(float(values[index]), EPSILON)
        params['Xr1'] = params['reactance_ratio'] * params['Xs']
        return params

    best_params = build_params(best_values)
    best_error = single_cage_fit_error(lab_data, best_params)
    iteration = 0

    while step > math.log(1.0025) and iteration < max_iter:
        improved = False
        for index in range(len(variable_keys)):
            for direction in (1.0, -1.0):
                trial_values = np.array(best_values, copy=True)
                trial_values[index] *= math.exp(direction * step)
                trial_params = build_params(trial_values)
                trial_error = single_cage_fit_error(lab_data, trial_params)
                if trial_error < best_error:
                    best_values = trial_values
                    best_params = trial_params
                    best_error = trial_error
                    improved = True
        if not improved:
            step *= 0.5
        iteration += 1

    return best_params, best_error


def single_cage_summary(lab_data, params):
    sync_speed_rpm = synchronous_speed(lab_data['frequency_hz'], lab_data['pole_count'])
    full_load_slip = (sync_speed_rpm - lab_data['rated_speed_rpm']) / sync_speed_rpm
    full_load_slip = min(max(full_load_slip, 1e-4), 1.0)

    rated_perf = single_cage_performance(lab_data, params, full_load_slip)
    blocked_perf = single_cage_performance(lab_data, params, 1.0)

    slips = np.linspace(1e-4, 1.0, 2000)
    torque_values = np.array([single_cage_performance(lab_data, params, slip)['torque_nm'] for slip in slips])
    breakdown_index = int(np.argmax(torque_values))
    breakdown_perf = single_cage_performance(lab_data, params, float(slips[breakdown_index]))
    full_load_torque = rated_torque(lab_data['rated_power_w'], lab_data['rated_speed_rpm'])

    return {
        'sync_speed_rpm': sync_speed_rpm,
        'full_load_slip': full_load_slip,
        'rated': rated_perf,
        'locked_rotor': blocked_perf,
        'breakdown': breakdown_perf,
        'rated_torque_nm': full_load_torque,
        'targets': {
            'sf': full_load_slip,
            'rated_eff': rated_perf['efficiency'],
            'rated_pf': rated_perf['power_factor'],
            'T_b': breakdown_perf['torque_nm'] / full_load_torque,
            'T_lr': blocked_perf['torque_nm'] / full_load_torque,
            'I_lr': blocked_perf['current_a'] / lab_data['rated_current_a'],
        }
    }


def single_cage_curves(lab_data, params):
    slips = np.linspace(1.0, 1e-4, 500)
    points = [single_cage_performance(lab_data, params, float(slip)) for slip in slips]
    return {
        'slip': np.array([point['slip'] for point in points]),
        'speed_rpm': np.array([point['speed_rpm'] for point in points]),
        'torque_nm': np.array([point['torque_nm'] for point in points]),
        'current_a': np.array([point['current_a'] for point in points]),
    }


def double_cage_performance(lab_data, params, slip, line_voltage_v=None):
    slip = min(max(slip, 1e-4), 1.0)
    line_voltage = lab_data['rated_voltage_v'] if line_voltage_v is None else line_voltage_v
    phase_voltage = line_voltage / math.sqrt(3.0)
    sync_speed_rpm = synchronous_speed(lab_data['frequency_hz'], lab_data['pole_count'])
    omega_sync = 2.0 * math.pi * sync_speed_rpm / 60.0

    zs = complex(params['Rs'], params['Xs'])
    ym = complex(1.0 / params['Rc'], -1.0 / params['Xm'])
    zr1 = complex(params['Rr1'] / slip, params['Xr1'])
    zr2 = complex(params['Rr2'] / slip, params['Xr2'])

    y_parallel = ym + (1.0 / zr1) + (1.0 / zr2)
    z_parallel = 1.0 / y_parallel
    z_total = zs + z_parallel

    stator_current = phase_voltage / z_total
    air_gap_voltage = phase_voltage - stator_current * zs
    rotor_current_1 = air_gap_voltage / zr1
    rotor_current_2 = air_gap_voltage / zr2

    apparent_power = 3.0 * phase_voltage * np.conj(stator_current)
    input_power = float(np.real(apparent_power))
    reactive_power = abs(float(np.imag(apparent_power)))

    air_gap_power = 3.0 * (
        (abs(rotor_current_1) ** 2) * params['Rr1'] / slip +
        (abs(rotor_current_2) ** 2) * params['Rr2'] / slip
    )
    converted_power = air_gap_power * (1.0 - slip)
    shaft_power = max(converted_power - lab_data['fw_loss_w'], 0.0)
    torque_nm = air_gap_power / omega_sync

    current_a = abs(stator_current)
    apparent_magnitude = math.sqrt(3.0) * line_voltage * current_a
    power_factor = input_power / apparent_magnitude if apparent_magnitude > EPSILON else 0.0
    power_factor = min(max(power_factor, 0.0), 1.0)

    efficiency = shaft_power / input_power if input_power > EPSILON else 0.0
    efficiency = min(max(efficiency, 0.0), 1.0)

    return {
        'slip': slip,
        'speed_rpm': sync_speed_rpm * (1.0 - slip),
        'torque_nm': torque_nm,
        'current_a': current_a,
        'input_power_w': input_power,
        'reactive_power_var': reactive_power,
        'shaft_power_w': shaft_power,
        'power_factor': power_factor,
        'efficiency': efficiency,
    }


def double_cage_no_load_slip(lab_data, params, line_voltage_v=None):
    slips = np.geomspace(1e-4, 0.2, 600)
    best_slip = float(slips[0])
    best_error = None

    for slip in slips:
        point = double_cage_performance(lab_data, params, float(slip), line_voltage_v=line_voltage_v)
        error = abs(point['shaft_power_w'])
        if best_error is None or error < best_error:
            best_slip = float(slip)
            best_error = error

    return best_slip


def double_cage_lab_metrics(lab_data, params):
    no_load_slip = double_cage_no_load_slip(
        lab_data,
        params,
        line_voltage_v=lab_data['no_load_voltage_v'],
    )
    no_load = double_cage_performance(
        lab_data,
        params,
        no_load_slip,
        line_voltage_v=lab_data['no_load_voltage_v'],
    )
    blocked = double_cage_performance(
        lab_data,
        params,
        1.0,
        line_voltage_v=lab_data['blocked_voltage_v'],
    )
    starting = double_cage_performance(
        lab_data,
        params,
        1.0,
        line_voltage_v=lab_data['rated_voltage_v'],
    )
    sync_speed_rpm = synchronous_speed(lab_data['frequency_hz'], lab_data['pole_count'])
    rated_slip = (sync_speed_rpm - lab_data['rated_speed_rpm']) / sync_speed_rpm
    rated = double_cage_performance(lab_data, params, rated_slip)

    return {
        'no_load_current_a': no_load['current_a'],
        'no_load_power_w': no_load['input_power_w'],
        'blocked_current_a': blocked['current_a'],
        'blocked_power_w': blocked['input_power_w'],
        'starting_current_a': starting['current_a'],
        'rated_current_a': rated['current_a'],
        'rated_pf': rated['power_factor'],
        'rated_eff': rated['efficiency'],
        'rated_power_w': rated['shaft_power_w'],
    }


def estimate_double_cage_initial_parameters(lab_data):
    single = estimate_single_cage_parameters(lab_data)
    rc_guess = single['Rc'] if single['Rc'] is not None else 1e6
    return {
        'Rs': single['Rs'],
        'Xs': max(single['Xs'] * 1.4, EPSILON),
        'Xm': max(single['Xm'], EPSILON),
        'Rr1': max(single['Rr1'] * 0.35, EPSILON),
        'Xr1': max(single['Xr1'] * 2.9, EPSILON),
        'Rr2': max(single['Rr1'] * 1.85, EPSILON),
        'Xr2': max(single['Xr1'] * 0.8, EPSILON),
        'Rc': max(rc_guess, EPSILON),
    }


def double_cage_fit_error(lab_data, params):
    metrics = double_cage_lab_metrics(lab_data, params)
    weights = {
        'no_load_current_a': 1.0,
        'no_load_power_w': 1.0,
        'blocked_current_a': 0.5 if lab_data.get('starting_current_target_a', 0.0) > 0.0 else 1.0,
        'blocked_power_w': 0.5 if lab_data.get('starting_current_target_a', 0.0) > 0.0 else 1.0,
        'starting_current_a': 1.25,
        'rated_current_a': 1.25,
        'rated_pf': 1.0,
        'rated_eff': 1.0,
        'rated_power_w': 1.25,
    }
    targets = {
        'no_load_current_a': lab_data.get('no_load_current_target_a', lab_data['no_load_current_a']),
        'no_load_power_w': lab_data.get('no_load_power_target_w', lab_data['no_load_power_w']),
        'blocked_current_a': lab_data['blocked_current_a'],
        'blocked_power_w': lab_data['blocked_power_w'],
        'starting_current_a': lab_data.get('starting_current_target_a', 0.0),
        'rated_current_a': lab_data.get('rated_current_target_a', lab_data['rated_current_a']),
        'rated_pf': lab_data.get('rated_pf_target', 0.0),
        'rated_eff': lab_data.get('rated_eff_target', 0.0),
        'rated_power_w': lab_data.get('rated_power_target_w', lab_data['rated_power_w']),
    }

    error = 0.0
    for key, target in targets.items():
        if target is None or target <= 0.0:
            continue
        delta = (metrics[key] - target) / max(abs(target), EPSILON)
        error += weights[key] * delta * delta

    if params['Rr2'] <= params['Rr1']:
        delta = (params['Rr1'] - params['Rr2']) / max(params['Rr2'], EPSILON)
        error += 10.0 * delta * delta
    if params['Xr1'] <= params['Xr2']:
        delta = (params['Xr2'] - params['Xr1']) / max(params['Xr1'], EPSILON)
        error += 10.0 * delta * delta

    return error


def fit_double_cage_parameters(lab_data, initial_params=None, max_iter=120):
    if initial_params is None:
        initial_params = estimate_double_cage_initial_parameters(lab_data)

    variable_keys = ['Xs', 'Xm', 'Rr1', 'Xr1', 'Rr2', 'Xr2', 'Rc']
    best_values = np.array([max(float(initial_params[key]), EPSILON) for key in variable_keys], dtype=float)
    step = math.log(1.4)

    def build_params(values):
        params = dict(initial_params)
        for index, key in enumerate(variable_keys):
            params[key] = max(float(values[index]), EPSILON)
        return params

    best_params = build_params(best_values)
    best_error = double_cage_fit_error(lab_data, best_params)
    iteration = 0

    while step > math.log(1.003) and iteration < max_iter:
        improved = False
        for index in range(len(variable_keys)):
            for direction in (1.0, -1.0):
                trial_values = np.array(best_values, copy=True)
                trial_values[index] *= math.exp(direction * step)
                trial_params = build_params(trial_values)
                trial_error = double_cage_fit_error(lab_data, trial_params)
                if trial_error < best_error:
                    best_values = trial_values
                    best_params = trial_params
                    best_error = trial_error
                    improved = True
        if not improved:
            step *= 0.55
        iteration += 1

    return best_params, best_error


def double_cage_summary(lab_data, params):
    sync_speed_rpm = synchronous_speed(lab_data['frequency_hz'], lab_data['pole_count'])
    full_load_slip = (sync_speed_rpm - lab_data['rated_speed_rpm']) / sync_speed_rpm
    full_load_slip = min(max(full_load_slip, 1e-4), 1.0)

    rated_perf = double_cage_performance(lab_data, params, full_load_slip)
    blocked_perf = double_cage_performance(lab_data, params, 1.0)

    slips = np.linspace(1e-4, 1.0, 2000)
    torque_values = np.array([double_cage_performance(lab_data, params, slip)['torque_nm'] for slip in slips])
    breakdown_index = int(np.argmax(torque_values))
    breakdown_perf = double_cage_performance(lab_data, params, float(slips[breakdown_index]))
    full_load_torque = rated_torque(lab_data['rated_power_w'], lab_data['rated_speed_rpm'])

    return {
        'sync_speed_rpm': sync_speed_rpm,
        'full_load_slip': full_load_slip,
        'rated': rated_perf,
        'locked_rotor': blocked_perf,
        'breakdown': breakdown_perf,
        'rated_torque_nm': full_load_torque,
    }


def double_cage_curves(lab_data, params):
    slips = np.linspace(1.0, 1e-4, 500)
    points = [double_cage_performance(lab_data, params, float(slip)) for slip in slips]
    return {
        'slip': np.array([point['slip'] for point in points]),
        'speed_rpm': np.array([point['speed_rpm'] for point in points]),
        'torque_nm': np.array([point['torque_nm'] for point in points]),
        'current_a': np.array([point['current_a'] for point in points]),
    }


def load_point_summary(lab_data, performance_callback):
    rated_torque_nm = rated_torque(lab_data['rated_power_w'], lab_data['rated_speed_rpm'])
    load_points = []

    # Scan the full slip range to locate the breakdown slip (peak torque).
    # Operating points must be in the stable region (s < s_breakdown).
    scan_slips = np.linspace(1e-4, 0.9999, 5000)
    scan_torques = [performance_callback(float(s))['torque_nm'] for s in scan_slips]
    breakdown_idx = int(np.argmax(scan_torques))
    stable_slips = scan_slips[:breakdown_idx + 1]

    for fraction in [0.25, 0.50, 0.75, 1.00]:
        target_torque = fraction * rated_torque_nm
        best = None
        best_error = None

        for slip in stable_slips:
            point = performance_callback(float(slip))
            error = abs(point['torque_nm'] - target_torque)
            if best_error is None or error < best_error:
                best = point
                best_error = error

        load_points.append({
            'load_fraction': fraction,
            'speed_rpm': best['speed_rpm'],
            'torque_nm': best['torque_nm'],
            'current_a': best['current_a'],
            'power_factor': best['power_factor'],
            'efficiency': best['efficiency'],
        })

    return load_points
