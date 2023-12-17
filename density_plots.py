import os
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
import pytensor
from pytensor.tensor import TensorVariable
from pymc.distributions.transforms import Interval
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
import string
from formulae import design_matrices
import itertools
from pytensor.printing import Print
from scipy.stats import gaussian_kde


# -------------------------------------------------------------------------------------------

def build_design_matrices(
    formula,
    df,
    query=None,
    drop=None,
):
    
    indices = np.arange(df.shape[0])
    if query:
        indices = df.query(query).index
        
    dm = design_matrices(formula, df)
    dm_common = None

    if dm.group is not None:
        colnames = []
        for group, vals in dm.group.slices.items():
            stop = vals.stop - vals.start
            val = list(range(0, stop))
            cols = [f"{group}[{i}]" for i in val]
            colnames.extend(cols)
        dm_groups = pd.DataFrame(dm.group.design_matrix, columns=colnames)
    
    if dm.common is not None:
        dm_common = dm.common.as_dataframe()
    
    dm = pd.concat([dm_common, dm_groups], axis=1)
    
    if drop:
        dm.drop(list(set(drop).intersection(dm.columns)),
                    axis=1, inplace=True)
    
    return dm.iloc[indices]




def plot_density_with_ref_and_rope(data, reference=0, rope=None, ax=None, 
                                   add_legend=False, remove_yticks=True, remove_xaxis=False,
                                   title=None, xlabel=None, xtick_fontsize=None,
                                   title_fontsize=None, subplot_title_fontsize=None,
                                   percent_region=0.05):
    
    # If no axis is provided, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Generate kernel density estimate
    density = gaussian_kde(data)
    xs = np.linspace(min(data) - 1, max(data) + 1, 1000)
    ys = density(xs)
    
    # Plot the density
    ax.plot(xs, ys, label='Density', color='blue')
    ax.fill_between(xs, 0, ys, color='blue', alpha=0.25)  # Light fill for entire density
    
    # Remove y-ticks if specified
    if remove_yticks:
        ax.set_yticks([])
    
    # Remove x-axis if specified
    if remove_xaxis:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
        ax.axhline(y=0, color='black', linewidth=1.2)  # Add bottom line to close the density plot
    else:
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_position(('data', 0))  # Ensures no gap between the density and x-axis
    
    # Remove top and right axis borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set subplot title
    if title is not None:
        ax.set_title(title, fontsize=subplot_title_fontsize)
    
    # Set x-label
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xtick_fontsize)
    
    # Set x-tick font size
    if xtick_fontsize is not None:
        ax.tick_params(axis='x', labelsize=xtick_fontsize)
    
    # Add legend only to the first subplot if specified
    if add_legend and (ax.is_first_col() if hasattr(ax, 'is_first_col') else True):
        ax.legend(loc='upper right', fontsize=xtick_fontsize)
    
    # If ROPE is provided, compute and show the probability within ROPE
    if rope is not None:
        lower_ref_density = density(rope[0])[0]  # Get the scalar value
        upper_ref_density = density(rope[1])[0]  # Get the scalar value
        ymax_lower = lower_ref_density / max(ys)
        ymax_upper = upper_ref_density / max(ys)
        ax.axvline(x=rope[0], ymax=ymax_lower, color='green', linestyle='--')
        ax.axvline(x=rope[1], ymax=ymax_upper, color='green', linestyle='--')
        ax.fill_between(xs, 0, ys, where=(xs >= rope[0]) & (xs <= rope[1]), color='green', alpha=0.3)
        prob_in_rope = density.integrate_box_1d(rope[0], rope[1])
        ax.annotate(f'{prob_in_rope:.1%} in ROPE', 
                     xy=((rope[0]+rope[1])/2, max(lower_ref_density, upper_ref_density)/100), 
                     xytext=(0, 10), 
                     textcoords='offset points',
                     ha='center', va='bottom')
    
    # Show the reference line if provided
    if reference is not None:
        ref_density = density(reference)[0]  # Get the scalar value
        ymax = ref_density / max(ys)
        ax.axvline(x=reference, ymax=ymax, color='red', linestyle='--')
        ax.fill_between(xs, 0, ys, where=(xs <= reference), color='red', alpha=0.3)
        prob_below = density.integrate_box_1d(-np.inf, reference)
        prob_above = 1 - prob_below
        ax.annotate(f'{prob_below:.1%} below < {reference} < {prob_above:.1%} above', 
                     xy=(reference, ref_density/20), 
                     xytext=(0, 10), 
                     textcoords='offset points',
                     ha='center', va='bottom')
    
    if (add_legend is False) and (ax.get_legend() is not None):
        ax.get_legend().remove()

    # Ensure the y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    if percent_region is not None:
        y = ax.get_ylim()[1]*.01
        lower, upper = np.quantile(data, q=[percent_region, 1-percent_region])
        ax.hlines(xmin=lower, xmax=upper, y=y, color='black', linewidth=2)

    # Show the plot if we are not passing in an Axes object
    if ax is None:
        plt.show()
        
        
# -------------------------------------------------------------------------------------------


def plot_density(data, reference=0, rope=None, percent_region=0.05, remove_xaxis=True, **kwargs):
    
    # If no axis is provided, create one
    if "ax" not in kwargs:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        
    ax = sns.kdeplot(data, cut=0, color="k", lw=.6, **kwargs)
    
    # Retrieve the density line for ROPE and reference area shading
    density_line = ax.get_lines()
    line_idx = len(density_line) - 1
    xs = density_line[line_idx].get_xdata()
    ys = density_line[line_idx].get_ydata()
    
    xmin, xmax = xs.min(), xs.max()
    
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.hlines(y=0, xmin=xmin, xmax=xmax, color='black', linewidth=2.2)
    lower, upper = np.quantile(data, q=[percent_region, 1-percent_region])
    fill = ax.fill_between(xs, 0, ys, where=(xs >= lower) & (xs <= upper), alpha=0.25)  # Light fill for entire density
    colour = fill.get_facecolor()
    ax.fill_between(xs, 0, ys, where=(xs <= lower) & (xs >= upper), alpha=0.05, color=colour)  # Light fill for entire density
    
    if remove_xaxis:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
    else:
        ax.spines['bottom'].set_visible(True)
        ax.tick_params(axis='x', labelsize=8)
    ax.spines['bottom'].set_position(('data', 0))  # Ensures no gap between the density and x-axis
    
    # Show the reference line if provided
    if reference is not None:
        ref_density = (xs.max() - xs.min())/2  # Get the scalar value
        ymax = ys[np.argwhere(xs <= reference).max()]
        ax.vlines(x=reference, ymin=0, ymax=ymax, linestyle='dotted', lw=1)
        prob_below = (xs <= 10).sum() / len(xs)
        prob_above = 1 - prob_below
        ax.annotate(f'{prob_below:.1%} < {reference} < {prob_above:.1%}', 
                     xy=(ref_density, ymax/20), 
                     xytext=(0, 10), 
                     textcoords='offset points',
                     ha='center', va='bottom')
        
    ax.set_xlim(xmin, xmax)
    
    if "ax" not in kwargs:
        plt.show()
        
# -------------------------------------------------------------------------------------------


high_level_cols = [

]

exclude = [

]

queries = []
queries_idx = []
subplot_titles = []
query_label = defaultdict(str)
query_map = defaultdict(list)

for col in high_level_cols:
    col_list = col.split("|")
    values = df[col_list].value_counts().index.to_list()
    if len(col_list) == 1:
        values = [val[0] for val in values]
        for val in values:
            new_query = f"({col} == {val})"
            query_idx = df.query(new_query).index.tolist()
            queries.append(new_query)
            queries_idx.append(query_idx)
            query_map[col].append(new_query)
            query_label[new_query] = f"{col}={val}"
    else:
        col1, col2 = col_list
        for val in values:
            query1 = f"({col1} == {val[0]})"
            query2 = f"({col2} == {val[1]})"
            if query2 in exclude:
                continue
            else:
                new_query = query1 + " & " + query2
                query_idx = df.query(new_query).index.tolist()
                queries.append(new_query)
                queries_idx.append(query_idx)
                query_map[f"{col1}|{col2}={val[1]}"].append(new_query)
                query_label[new_query] = f"{col}={val[0]},{col2}={val[1]}"

# check for errors
for idx, query in zip(queries_idx, queries):
    assert (df.iloc[idx] == df.query(query)).values.all()


# -------------------------------------------------------------------------------------------


rng = np.random.default_rng(seed=88)
n_samples = 200
nrows = len(query_map.keys())
ylabels = list(query_map.keys())

fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(1 + 4, nrows/1.7))

for i, item in enumerate(zip(query_map.items())):
    col, q = item[0]

i = 0
for col, query in query_map.items():
    
    for q in query:
    
        indices = df.query(q).index
        random_indices = rng.choice(indices, size=n_samples, replace=True)
        values = idata.posterior_predictive.likelihood.sel(i=random_indices).mean(["chain", "draw"]).values
        val = values[...,0]
        plot_density(val, ax=ax[i,0], reference=None)
        plot_density(val, ax=ax[i,1], reference=None)
        ax[i,0].set_ylabel(ylabels[i], fontsize=8, rotation=0, labelpad=80)
        ax[0,0].set_title("y1", fontsize=9)
    i += 1

xmin, xmax = np.inf, -np.inf
for axis in ax[:,0]:
    xmin, xmax = axis.get_xlim()
    if xmin < xmin:
        xmin = xmin
    if xmax > xmax:
        xmax = xmax
    
# Apply the x-ticks to each subplot
xticks = [5, 10, 15, 20]
for axis in ax[:,0]:
    axis.set_xlim(xmin, xmax)
    
fig.tight_layout()

# -------------------------------------------------------------------------------------------

with model:
    svgd = pm.fit(
        n=30,
        method="svgd",        
        inf_kwargs=dict(n_particles=100, jitter=.1),
        obj_optimizer=pm.adamax(learning_rate=0.1),
    )
    idata = svgd.sample(100)
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    
    
# -------------------------------------------------------------------------------------------

"""
minibatches

svgd

remove sigma multiplier?

use actual observed fired data for hit inputs

for pm.graphviz, change ConstantData to MutableData and add in inputs_f inputs_h. To run the model and save it, rbcross, and have all other variables as MutableData

change pm.Constraint to have pt.log(0) !!
add jitter=.1 in SVGD!!!
newer versions of pymc has issues with mutable data. pymc 5.9.0 works fine.

Use ideas from the paper Accommodating binary and count variables in mediation: A case for conditional indirect effects. 
Explain conditional indirect effects/associations, and conditional total effects/associations. use similar diagram but tailor it to the study.

statistical rethinking for plots
"""

