"""Herten example."""
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from mpl_toolkits import axes_grid1
import seaborn as sns
import numpy as np
import gstools as gs


# some plotting definitions
s_size_2d = 1.5
s_size_1d = 10.0


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


# load the Herten aquifer #####################################################

print("Loading data")
herten_T = np.loadtxt(Path("../data/herten_transmissivity.gz"))
dim, origin, spacing = np.loadtxt(Path("../data/grid_dim_origin_spacing.txt"))

# create a structured grid on which the data is defined
x_s = np.arange(origin[0], origin[0] + dim[0] * spacing[0], spacing[0])
y_s = np.arange(origin[1], origin[1] + dim[1] * spacing[1], spacing[1])

# imshow plotting settings
extent = [min(x_s), max(x_s), min(y_s), max(y_s)]
kw1 = dict(origin="lower", extent=extent)

idx_x, idx_y = np.mgrid[40:170:10, 40:170:10]
idx_x, idx_y = idx_x.flatten(), idx_y.flatten()

obs_x = x_s[idx_x]
obs_y = y_s[idx_y]
obs_val = herten_T[idx_x, idx_y].reshape(-1)

fig, ax = plt.subplots()
im = ax.imshow(herten_T.T, **kw1)
ax.set_xlabel(r"Long. Direction $x$ / m")
ax.set_ylabel(r"Trans. Direction $y$ / m")
ax.scatter(obs_x, obs_y, color="k", s=s_size_2d)
cbar = add_colorbar(im)
cbar.set_label(r"Transmissivity $T$ / m$^{2}$ s$^{-1}$")

fig.tight_layout()
fig.savefig(Path("../results/herten_T_obs.pdf"), dpi=300)
fig.show()

###############################################################################
# estimate the variogram on an unstructured grid ##############################
###############################################################################

print("Estimating variogram")

# assume the data to be log-normal distributed
norm = gs.normalizer.LogNormal()

bins = np.linspace(0, 7, 10)
bin_center, gamma = gs.vario_estimate(
    (obs_x, obs_y), obs_val, bins, normalizer=norm
)
# fit an exponential model
fit_model = gs.Exponential(dim=2)
fit = fit_model.fit_variogram(bin_center, gamma, nugget=False, return_r2=True)
ax = fit_model.plot(x_max=max(bin_center))
ax.scatter(bin_center, gamma)
ax.set_xlabel(r"Distance $r$ / m")
ax.set_ylabel(r"Variogram$")

fig = ax.get_figure()
fig.tight_layout()
fig.savefig(Path("../results/herten_variogram.pdf"), dpi=300)
fig.show()

print("Coefficient of determination of the fit R² = {:.3}".format(fit[2]))
print("semivariogram model (isotropic):")
print(fit_model)

###############################################################################
# creating a SRF from the Herten parameters ###################################
###############################################################################

ok = gs.krige.Ordinary(fit_model, (obs_x, obs_y), obs_val, normalizer=norm)
csrf = gs.CondSRF(ok)

herten_ens = []
master_seed = gs.random.MasterRNG(20060906)
for i in range(20):
    seed = master_seed()
    print(f"{i:3}: Calculating conditioned SRF with seed {seed:6}")
    herten_ens.append(csrf.structured((x_s, y_s), seed=seed))
herten_ens = np.array(herten_ens)

###############################################################################
# Plotting ####################################################################
###############################################################################

vmin = min(np.min(herten_T), np.min(herten_ens))
vmax = max(np.max(herten_T), np.max(herten_ens))
kw2 = dict(vmin=vmin, vmax=vmax, **kw1)
c = sns.color_palette()

# compare Original field with one realization #################################

# Create a Rectangle patch
rect_dim = ((min(obs_x), min(obs_y)), np.ptp(obs_x), np.ptp(obs_y))
rect = pat.Rectangle(*rect_dim, edgecolor='k', facecolor='none')

fig, ax = plt.subplots(1)
im = ax.imshow(herten_T.T, **kw2)
cbar = add_colorbar(im)
cbar.set_label(r"$T$ / m s$^{-1}$")
ax.set_xlabel(r"$x$ / m")
ax.set_ylabel(r"$y$ / m")
ax.add_patch(copy.copy(rect))

fig.tight_layout()
fig.savefig(Path("../results/2d_herten_conditioned_0.pdf"), dpi=300)
fig.show()

fig, ax = plt.subplots(1)
im = ax.imshow(herten_ens[0].T, **kw2)
cbar = add_colorbar(im)
cbar.set_label(r"$T$ / m s$^{-1}$")
ax.set_xlabel(r"$x$ / m")
ax.set_ylabel(r"$y$ / m")
ax.add_patch(copy.copy(rect))

fig.tight_layout()
fig.savefig(Path("../results/2d_herten_conditioned_1.pdf"), dpi=300)
fig.show()

# plot absolute difference ####################################################

fig, ax = plt.subplots()
im2 = ax.imshow(np.abs(herten_T.T - herten_ens[0].T), cmap="YlOrRd", **kw1)
ax.set_aspect("equal")
cbar = add_colorbar(im2)
cbar.set_label(r"Absolute Difference $T$ / m s$^{-1}$")
ax.set_xlabel(r"$x$ / m")
ax.set_ylabel(r"$y$ / m")

fig.tight_layout()
fig.savefig(Path("../results/2d_herten_difference.pdf"), dpi=300)
fig.show()

# plot cross-section and standard deviation ###################################

trans_idx_y = 50
trans_idx_x = idx_x[np.argwhere(idx_y == trans_idx_y)]
K_cond_std = np.std(herten_ens[:, :, trans_idx_y], axis=0)

fig, ax = plt.subplots()

# uncomment this to plot the single realisations
# for i in range(len(herten_ens)):
#     ax.plot(x_s, herten_ens[i][:, trans_idx_y], color=c[0], alpha=0.3)

ax.plot(x_s, herten_T[:, trans_idx_y], color=c[0], linewidth=2.0)
ax.fill_between(
    x_s,
    herten_T[:, trans_idx_y] - K_cond_std,
    herten_T[:, trans_idx_y] + K_cond_std,
    alpha=0.6,
)
ax.scatter(
    x_s[trans_idx_x],
    herten_T[trans_idx_x, trans_idx_y],
    color="k",
    s=s_size_1d,
    zorder=10,
)
ax.set_ylim(0.3, 0.65)
ax.set_ylabel(r"Transmissivity $T$ / m s$^{-1}$")
ax.set_xlabel(r"Distance $x$ / m")

fig.tight_layout()
fig.savefig(Path("../results/1d_herten_conditioned.pdf"), dpi=300)
fig.show()
