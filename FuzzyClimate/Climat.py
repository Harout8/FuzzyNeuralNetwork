import numpy as np
import skfuzzy as fuzz
import matplotlib
from skfuzzy import control as ctrl
import time


# New Antecedent objects hold universe variables and membership functions
temperature = ctrl.Antecedent(np.arange(0, 50, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 50, 1), 'humidity')
requestedTemperature = ctrl.Antecedent(np.arange(0, 50, 1), 'requestedTemperature')
roomVolume = ctrl.Antecedent(np.arange(0, 50, 1), 'roomVolume')

# New Consequent objects hold universe variables and membership functions
toCool =  b.Consequent(np.arange(0, 30, 1), 'toCool')
toWarm = ctrl.Consequent(np.arange(0, 30, 1), 'toWarm')


# Auto-membership function population is possible with .automf(3, 5, or 7)
temperature.automf(3)
humidity.automf(3)
requestedTemperature.automf(3)
roomVolume.automf(3)


# Custom membership functions can be built interactively with a familiar, Pythonic API
toCool['aLittleBit'] = fuzz.trimf(toCool.universe, [0, 3, 6])
toCool['cooler'] = fuzz.trimf(toCool.universe, [6, 9, 12])
toCool['strongly'] = fuzz.trimf(toCool.universe, [12, 20, 30])

toWarm['aLittleBit'] = fuzz.trimf(toWarm.universe, [0, 3, 6])
toWarm['warmer'] = fuzz.trimf(toWarm.universe, [6, 9, 12])
toWarm['strongly'] = fuzz.trimf(toWarm.universe, [12, 20, 30])

temperature['average'].view()
humidity.view()
requestedTemperature.view()
roomVolume.view()

toCool.view()
toWarm.view()


# Fuzzy rules
rule1 = ctrl.Rule(temperature['poor'] | requestedTemperature['average'] & (humidity['average'] | roomVolume['average']), toWarm['warmer'])
rule2 = ctrl.Rule(temperature['poor'] | requestedTemperature['good'] & (humidity['average'] | roomVolume['average']), toWarm['strongly'])
rule3 = ctrl.Rule(temperature['average'] | requestedTemperature['good'] & (humidity['average'] | roomVolume['average']), toWarm['warmer'])
rule4 = ctrl.Rule(temperature['average'] | requestedTemperature['poor'] & (humidity['average'] | roomVolume['average']), toCool['cooler'])
rule5 = ctrl.Rule(temperature['good'] | requestedTemperature['poor'] & (humidity['average'] | roomVolume['average']), toCool['strongly'])
rule6 = ctrl.Rule(temperature['good'] | requestedTemperature['average'] & (humidity['average'] | roomVolume['average']), toCool['cooler'])

rule1.view()


# Control System Creation and Simulation
climat_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
climat = ctrl.ControlSystemSimulation(climat_ctrl)


# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
climat.input['temperature'] = 10
climat.input['humidity'] = 25
climat.input['requestedTemperature'] = 20
climat.input['roomVolume'] = 25

# Crunch the numbers
climat.compute()

# Visualize
print(climat.output['toWarm'])
toWarm.view(sim=climat)

print(climat.output['toCool'])
toCool.view(sim=climat)

#input('\n Нажмите Enter ==>  ')

time.sleep(100)
#while(True):

    #print("...")