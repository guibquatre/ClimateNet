#!/usr/bin/python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pathlib
import logging
import numpy as np
import functools

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('log_file',
    help='Path to a log file produced by a run of the simulation with the'
        ' analytic.json configuration file'
    )
args = argument_parser.parse_args()

log_file_path = pathlib.Path(args.log_file)
if not log_file_path.exists():
    logging.error(f'{log_file_path} does not exists')
    exit(code=1)

data = { 'Velocity' : [], 'Position' : [] }

with open(log_file_path) as log_file:
    current_line = log_file.readline()
    while current_line != '':
        if (
                not current_line.startswith('Velocity')
                and not current_line.startswith('Position')
            ):
            current_line = log_file.readline()
            continue

        data_type, number_points = current_line.strip('\n').split()
        number_points = int(number_points)
        vector = np.array([ 0., 0., 0. ])
        for _ in range(number_points):
            log_file.readline() # Skip the line which contains v*
            vector += np.array([
                float(log_file.readline()),
                float(log_file.readline()),
                float(log_file.readline())
            ])
        vector /= 4
        # We only take the last coordinate which represents the height.
        if data_type == 'Velocity':
            data[data_type].append(np.linalg.norm(vector))
        else:
            data[data_type].append(vector[2])
        current_line = log_file.readline()

def getTimeOfImpact(gravity, support_height, initial_position):
    return np.sqrt(2 * (initial_position - support_height) / gravity)

def getFreeFallVelocity(gravity, t):
    return gravity * t

def getVelocityBeforeImpact(gravity, support_height, initial_position):
    return getFreeFallVelocity(
            gravity,
            getTimeOfImpact(gravity, support_height, initial_position)
        )

def getVelocityAfterImpact(
        gravity,
        support_height,
        support_angle,
        initial_position,
        friction_coeffcient
    ):
    return (
            getVelocityBeforeImpact(gravity, support_height, initial_position)
            * (
                np.cos(support_angle)
                - np.sin(support_angle) * friction_coeffcient
            )
        )

def getAccelerationAfterImpact(gravity, support_angle, friction_coeffcient):
    return (
            gravity
            * (
                np.sin(support_angle)
                - np.cos(support_angle) * friction_coeffcient
            )
        )


def getAnalyticalVelocity(gravity, support_height, support_angle,
        initial_position, friction_coeffcient, t):
    t_impact = getTimeOfImpact(gravity, support_height, initial_position)
    if t < t_impact:
        return getFreeFallVelocity(gravity, t)
    else:
        velocity_after_impact = getVelocityAfterImpact(
                gravity,
                support_height,
                support_angle,
                initial_position,
                friction_coeffcient
            )
        acceleration_after_impact = getAccelerationAfterImpact(
                gravity,
                support_angle,
                friction_coeffcient
            )


        return (
                velocity_after_impact
                + acceleration_after_impact * (t - t_impact)
            )

def getAnalyticalPosition(
        gravity,
        support_height,
        support_angle,
        initial_position,
        friction_coeffcient,
        t
    ):
    t_impact = getTimeOfImpact(gravity, support_height, initial_position)
    if t < t_impact:
        return initial_position - .5 * gravity * (t ** 2)
    else:
        velocity_after_impact = getVelocityAfterImpact(
                gravity,
                support_height,
                support_angle,
                initial_position,
                friction_coeffcient
            )

        acceleration_after_impact = getAccelerationAfterImpact(
                gravity,
                support_angle,
                friction_coeffcient
            )

        return (
                support_height
                - np.sin(support_angle) * (
                    velocity_after_impact * (t - t_impact)
                    + .5 * acceleration_after_impact * ((t - t_impact) ** 2)
                )
            )

steps = np.linspace(0, 0.8, num=1000)
gravity = 9.81
support_angle = np.pi / 4
support_height = 0.0
initial_position = 0.2
friction_coeffcient = 0.1

getVelocity = functools.partial(
        getAnalyticalVelocity,
        gravity,
        support_height,
        support_angle,
        initial_position,
        friction_coeffcient
    )

getPosition = functools.partial(
        getAnalyticalPosition,
        gravity,
        support_height,
        support_angle,
        initial_position,
        friction_coeffcient
    )

analytical_velocities = np.array(list(map(getVelocity, steps)))
analytical_positions = np.array(list(map(getPosition, steps)))

simulation_linewidth = 7
analytical_linewidth = 2
simulation_color = 'dodgerblue'
analytical_color = 'red'

plt.subplot(121)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m.s^{-1})')
plt.plot(
        np.arange(len(data['Velocity'])) * 1e-3,
        data['Velocity'],
        linewidth=simulation_linewidth,
        color=simulation_color
    )
plt.plot(
        steps,
        analytical_velocities,
        linewidth=analytical_linewidth,
        color=analytical_color
    )
plt.legend(['Ours', 'Analytical'], fontsize=20, loc=2)
plt.subplot(122)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.plot(
        np.arange(len(data['Position'])) * 1e-3,
        data['Position'],
        linewidth=simulation_linewidth,
        color=simulation_color
    )
plt.plot(
        steps,
        analytical_positions,
        linewidth=analytical_linewidth,
        color=analytical_color
    )
plt.savefig('Figure5/plots.png')
