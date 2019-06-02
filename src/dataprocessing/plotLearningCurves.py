import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn')

baseline_metrics = []
baseline_log = open('../../output/logs/baseline.txt')
for line in baseline_log:
    m = re.search('{.*}', line)

    if m:
        metric = eval(m.group())
        baseline_metrics.append(metric)

two_more_metrics = []
two_more_log = open('../../output/logs/two_more.txt')
for line in two_more_log:
    m = re.search('{.*}', line)

    if m:
        metric = eval(m.group())
        two_more_metrics.append(metric)


double_metrics = []
double_log = open('../../output/logs/double.txt')
for line in double_log:
    if 'Training stage: 1' in line:
        break
for line in double_log:
    m = re.search('{.*}', line)

    if m:
        metric = eval(m.group())
        double_metrics.append(metric)

baseline_df = pd.DataFrame(baseline_metrics)
baseline_NCC = baseline_df[baseline_df['metric'] == 'NCC']
baseline_val_NCC = baseline_df[baseline_df['metric'] == 'Mean Validation NCC']

two_more_df = pd.DataFrame(two_more_metrics)
two_more_NCC = two_more_df[two_more_df['metric'] == 'NCC']
two_more_val_NCC = two_more_df[two_more_df['metric'] == 'Mean Validation NCC']

double_df = pd.DataFrame(double_metrics)
double_df['step'] -= double_df['step'].min()
double_df['step'] += two_more_df['step'].max() + 1
double_NCC = double_df[double_df['metric'] == 'NCC']
double_val_NCC = double_df[double_df['metric'] == 'Mean Validation NCC']

fig, ax = plt.subplots(ncols=2, figsize=(10, 6), sharey=True)
ax[0].plot(baseline_NCC['step'], baseline_NCC['value'])
ax[0].plot(baseline_NCC['step'], baseline_NCC['value'].rolling(window=100).mean())
ax[0].plot(baseline_val_NCC['step'], baseline_val_NCC['value'])
ax[0].set_xlabel('Step')
ax[0].set_ylabel('NCC')

ax[1].plot(two_more_NCC['step'], two_more_NCC['value'])
ax[1].plot(two_more_NCC['step'], two_more_NCC['value'].rolling(window=100).mean())
ax[1].plot(two_more_val_NCC['step'], two_more_val_NCC['value'])
ax[1].legend(['Training', 'Training (100 step moving avg.)', 'Validation'])
ax[1].set_xlabel('Step')
ax[1].set_ylabel('NCC')
plt.savefig('learning_curves_single.eps', dpi=150)
plt.close()

fig, ax = plt.subplots()
ax.plot(two_more_NCC['step'], two_more_NCC['value'], color='C0', label='Training')
ax.plot(double_NCC['step'], double_NCC['value'], color='C0', label='_nolegend_')
ax.plot(two_more_NCC['step'], two_more_NCC['value'].rolling(window=100).mean(), color='C1', label='Training (100 step moving avg.)')
ax.plot(double_NCC['step'], double_NCC['value'].rolling(window=100).mean(), color='C1', label='_nolegend_')
ax.plot(two_more_val_NCC['step'], two_more_val_NCC['value'], color='C2', label='Validation')
ax.plot(double_val_NCC['step'], double_val_NCC['value'], color='C2', label='_nolegend_')
ax.axvline(x=two_more_NCC['step'].max(), linestyle='--', color='grey')
ax.legend()
ax.set_xlabel('Step')
ax.set_ylabel('NCC')
plt.savefig('learning_curves_double.eps', dpi=150)
plt.close()
