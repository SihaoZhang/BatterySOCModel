# BatterySOCModel
This model is used to estimate the state of charge(SOC) of battery based on NN and Kalman filter.

## Feature Engineering
Test data comes from the real program, which contains multiple features like battery voltage, current, SOC measured by the formulas and the environment temperature of different groups. 

* According to the battery function, divided the raw data into working conditions, performed prediction and evaluation according to the charging and discharging process.

* Using average smoothing method to eliminate some NaN points and outliers.

* Selecting one of the groups to analysis, and combining features such as maxTemperatureBatteryValue and minTemperatureBatteryValue into BatteryTemperature with their average.

* According to the principle of battery charge and discharge, extracting the charge and discharge current, charge and discharge voltage, and ambient temperature of the battery as features;

## RBF Model
Selected the discharge process characteristic data of the two sets of batteries and training with the RBF neural network to obtain a neural network(NN) model for predicting the SOC value of the battery.

## Kalman Filter
The NN model is used to obtain the predicted SOC, and the difference between the predicted value and the observed SOC is incorporated into the new measurement error as a past measurement error to estimate the future error, thereby obtaining an estimation of the SOC of the battery.