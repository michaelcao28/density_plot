# significantly improves image quality
%config InlineBackend.figure_format = 'retina'

# fig.tight_layout() is discouraged now. constrained_layout() works better
# to set it as default run:
plt.rcParams['figure.constrained_layout.use'] = True

# see styles and change style
plt.style.available 
plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])

# view and change colour cycle
from cycler import cycler
plt.rcParams['axes.prop_cycle']
current_cycler = plt.rcParams['axes.prop_cycle']
colors = [c['color'] for c in current_cycler]
new_color_order = colors[::-1] # reversed order

# change colour cycle globally
plt.rcParams['axes.prop_cycle'] = cycler(color=new_color_order)

# change colour cycle only for a particular plot
ax.set_prop_cycle(cycler(color=new_color_order))


# subplots using gridspec. This trick is not in api docs.
# Trick: treat each grid like the axes just like in plt.subplots()
fig = plt.figure(figsize=(l,h))
gs = fig.add_gridspec(
    nrows=nrows, 
    ncols=ncols, 
    figure=fig, 
    width_ratios=[.6, 1, 1], 
    height_ratios=[.1] + [1]*(nrows-1)
)
axes = gs.subplots() # shape (nrows x ncols)
