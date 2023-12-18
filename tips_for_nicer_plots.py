# significantly improves image quality
%config InlineBackend.figure_format = 'retina'

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
