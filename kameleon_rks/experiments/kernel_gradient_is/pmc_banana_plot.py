from matplotlib.lines import Line2D
import os

from kameleon_rks.experiments.kernel_gradient_is.pmc_banana import result_fname
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# from kameleon_rks.experiments.latex_plot_init import plt
result_fname_base = os.path.splitext(result_fname)[0]

sampler_names = [
                 'StaticMetropolis',
                 'AdaptiveMetropolis',
                 'StaticLangevin',
                 'AdaptiveLangevin',
                 'OracleKernelAdaptiveLangevin',
                 'KernelAdaptiveLangevin',
                 
                 ]
fields = [
            'rmse_mean',
            'rmse_cov',
            'mmd',
          ]

sampler_plot_names = {
                  'StaticMetropolis': 'SM',
                  'AdaptiveMetropolis': 'AM',
                  'StaticLangevin': 'SGIS',
                  'AdaptiveLangevin': 'GIS',
                  'OracleKernelAdaptiveLangevin': 'OKGIS',
                  'KernelAdaptiveLangevin': 'KGIS',
                  
                  }
sampler_colors = {
                  'StaticMetropolis': 'blue',
                  'AdaptiveMetropolis': 'red',
                  'StaticLangevin': 'green',
                  'AdaptiveLangevin': 'yellow',
                  'OracleKernelAdaptiveLangevin': 'magenta',
                  'KernelAdaptiveLangevin': 'black',
                  
                  }
field_plot_names = {
                    'mmd': 'MMD to benchmark sample',
                    'rmse_mean': 'RMSE mean',
                    'rmse_cov': 'RMSE covariance',
                    }

y_scales = {
            }


def kwargs_gen(**kwargs):
    return kwargs

conditions = kwargs_gen(
                          D=2,
                          bananicity=0.03,
                          V=100,
                          num_iter_per_particle=200,
                        )

# x-axis of plot
x_field = 'population_size'
x_field_values = []

df = pd.read_csv(result_fname, index_col=0)

for field in fields:
    plt.figure()
    
    for sampler_name in sampler_names:
        # filter out desired entries
        mask = (df.sampler_name == sampler_name)
        if mask.sum() == 0:
            print "No entries for %s" % sampler_name
            assert()
        
        for k, v in conditions.items():
            mask &= (df[k] == v)
        current = df.loc[mask]
        
        # only use desired values of x_fields
        if len(x_field_values) > 0:
            current = current.loc[[True if x in x_field_values else False for x in current[x_field]]]
    
        # avoid empty case
        if len(current) == 0:
            continue
    
        # use ints on x-axis
        current[x_field] = current[x_field].astype(int)
        
        # x axis of plot
        x_values = np.sort(np.unique(current[x_field].values))
        x_values = x_values[~np.isnan(x_values)]
        
        values = np.array([current.loc[current[x_field] == x_value][field].values for x_value in x_values])
        averages = np.array([np.mean(arr) for arr in values])
        plt.plot(x_values, averages, '-', color=sampler_colors[sampler_name])
        
        lowers = np.array([np.percentile(arr, 30) for arr in values])
        uppers = np.array([np.percentile(arr, 70) for arr in values])
        
        lower_errors = np.abs(averages - lowers)
        upper_errors = np.abs(averages - uppers)
        
        plt.errorbar(x_values, averages, yerr=[lower_errors, upper_errors], color=sampler_colors[sampler_name])

        # print info on number of trials
        print field
        print("Average number of trials: %d" % int(np.round(current.groupby(x_field).apply(len).mean())))
        print(current.groupby(x_field).apply(len))

    lines = [ Line2D([0, 0], [0, 0], color=sampler_colors[sampler_name]) for sampler_name in sampler_names]
    plt.legend(lines, [sampler_plot_names[sampler_name] for sampler_name in sampler_names])
    plt.xlabel("Population size")
    plt.ylabel(field_plot_names[field])
    
    if field in y_scales:
        plt.yscale(y_scales[field])
    
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(result_fname_base + ".png")
    plt.savefig(result_fname_base + ".eps")
    
plt.show()
