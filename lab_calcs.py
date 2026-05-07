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


def estimate_single_cage_parameters(lab_data):
    rated_frequency = lab_data['frequency_hz']
    blocked_frequency = lab_data['blocked_frequency_hz']
    blocked_voltage_phase = lab_data['blocked_voltage_v'] / math.sqrt(3.0)
    blocked_current = lab_data['blocked_current_a']

    if blocked_current <= 0.0:
        raise ValueError('Blocked-rotor current must be greater than zero.')
    if blocked_frequency <= 0.0:
        raise ValueError('Blocked-rotor frequency must be greater than zero.')

    rs = lab_data['blocked_resistance_ohm']
    z_blocked = blocked_voltage_phase / blocked_current
    r_total = lab_data['blocked_power_w'] / (3.0 * blocked_current ** 2)
    rr1 = max(r_total - rs, EPSILON)

    x_total_test = math.sqrt(max(z_blocked ** 2 - r_total ** 2, EPSILON))
    x_total_nominal = x_total_test * rated_frequency / blocked_frequency
    xs = x_total_nominal / 2.0
    xr1 = x_total_nominal / 2.0

    no_load_voltage_phase = lab_data['no_load_voltage_v'] / math.sqrt(3.0)
    no_load_current = lab_data['no_load_current_a']
    rs_no_load = correct_resistance(rs, lab_data['blocked_temp_c'], lab_data['no_load_temp_c'])
    no_load_copper_loss = 3.0 * no_load_current ** 2 * rs_no_load
    core_loss = max(lab_data['no_load_power_w'] - lab_data['fw_loss_w'] - no_load_copper_loss, EPSILON)

    rc = 3.0 * no_load_voltage_phase ** 2 / core_loss
    in_phase_current = no_load_voltage_phase / rc
    magnetizing_current = math.sqrt(max(no_load_current ** 2 - in_phase_current ** 2, EPSILON))
    xm = no_load_voltage_phase / magnetizing_current

    return {
        'Rs': rs,
        'Xs': xs,
        'Xm': xm,
        'Rr1': rr1,
        'Xr1': xr1,
        'Rc': rc,
    }


def single_cage_performance(lab_data, params, slip):
    slip = min(max(slip, 1e-4), 1.0)
    line_voltage = lab_data['rated_voltage_v']
    phase_voltage = line_voltage / math.sqrt(3.0)
    sync_speed_rpm = synchronous_speed(lab_data['frequency_hz'], lab_data['pole_count'])
    omega_sync = 2.0 * math.pi * sync_speed_rpm / 60.0

    zs = complex(params['Rs'], params['Xs'])
    ym = complex(1.0 / params['Rc'], -1.0 / params['Xm'])
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


def load_point_summary(lab_data, performance_callback):
    rated_torque_nm = rated_torque(lab_data['rated_power_w'], lab_data['rated_speed_rpm'])
    load_points = []
    trial_slips = np.linspace(1e-4, 0.5, 3000)

    for fraction in [0.25, 0.50, 0.75, 1.00]:
        target_torque = fraction * rated_torque_nm
        best = None
        best_error = None

        for slip in trial_slips:
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
