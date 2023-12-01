#!/usr/bin/python3
import sys
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':

    filename = sys.argv[1]
    stepToRead = [50, 100, 200, 300] 


    current_step = 0
    current_iteration = 0

    alart_curnier = {
        'TangentialAlartCurnier' : [],
        'NormalAlartCurnier' : [],
        'SelfTangentialAlartCurnier' : [],
        'SelfNormalAlartCurnier' : []
        }

    readStep = False
    for line in open(filename, 'r'):
        data = line.split(' ')
        if data[0] == 'Time':
            current_step = int(data[2])
            current_iteration = 0
            for _, v in alart_curnier.items():
                v.append([])
            readStep = (stepToRead is None) or (current_step in stepToRead)
            continue

        if data[0] in alart_curnier:
            if (readStep):
                alart_curnier[data[0]][-1].append(list(map(float, data[1:-1])))
            continue

    mean_alart_curnier = {}
    deviation_alart_curnier = {}
    max_alart_curnier = {}
    for k, data in alart_curnier.items():
        for step_data in data:
            mean_alart_curnier.setdefault(k, []).append([])
            deviation_alart_curnier.setdefault(k, []).append([])
            max_alart_curnier.setdefault(k, []).append([])
            for iteration_data in step_data:
                mean_alart_curnier[k][-1].append(sum(iteration_data) / len(iteration_data))
                deviation_alart_curnier[k][-1].append(math.sqrt(sum(map(lambda v: (v -
                    mean_alart_curnier[k][-1][-1])**2, iteration_data)) / len(iteration_data)))
                max_alart_curnier[k][-1].append(max(iteration_data))

    #print(mean_alart_curnier)
    #print(max_alart_curnier)

    #number_outlier = {}
    #number = {}
    #for k, data in mean_alart_curnier.items():
    #    number_outlier[k] = 0
    #    number[k] = 0
    #    for step_data in data:
    #        for iteration_data in step_data:
    #            number[k] += 1
    #            if iteration_data > 1e-3:
    #                number_outlier[k] += 1
    #
    #rate = {}
    #for k, data in number_outlier.items():
    #    rate[k] = data / number[k]
    #
    #print(rate)
    if stepToRead is None:
        stepToRead = [30, 45]

    legend = []
    normal_axis = plt.subplot(121)
    tangent_axis = plt.subplot(122)
    for ts in stepToRead:
        legend.append("t=" + str(ts * 5.e-3) + "s")
        normal_axis.plot(mean_alart_curnier['NormalAlartCurnier'][ts], linewidth=3)
        tangent_axis.plot(mean_alart_curnier['TangentialAlartCurnier'][ts], linewidth=3)

    normal_axis.set_xlabel('Local/Global iteration')
    normal_axis.set_ylabel('Normal error')
    normal_axis.set_yscale('log')

    tangent_axis.set_xlabel("Local/Global iteration")
    tangent_axis.set_ylabel("Tangential error")
    tangent_axis.set_xlim(right=100)
    tangent_axis.legend(legend, loc = 3)
    tangent_axis.set_yscale('log')

    plt.savefig('Figure6/plots.png')


