import matplotlib.pyplot as plt
import matplotlib.animation as animation
from jax import jit, vmap, lax
import jax.random as jr
import jax.numpy as jnp
import jax.scipy as jsp
from sklearn.cluster import KMeans
import numpy as np
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params
import blackjax
from scipy.signal import convolve2d
from jaxtyping import Array
from typing import Literal
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box
from matplotlib.animation import FuncAnimation
from blackjax.smc.tempered import TemperedSMCState

# functions to translate RGB to OKLAB and back


def linear_srgb_to_oklab(rgb: Array) -> Array:
    """Transforms a linear SRGB color vector to the corresponding representation in the OKLAB color space

    To apply to multiple vectors at one, use @jit

    Args:
        rgb (Array): SRGB color vector

    Returns:
        Array: OKLAB color vector
    """
    # transformation matrices
    M1 = jnp.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    M2 = jnp.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )

    # mapping
    lms = jnp.dot(M1, rgb)
    lms_ = jnp.cbrt(lms)
    oklab = jnp.dot(M2, lms_)

    return oklab


@jit
def oklab_to_linear_srgb(lab: Array) -> Array:
    """Transforms an OKLAB color vector to the corresponding representation in the SRGB color space

    To apply to multiple vectors at one, use @jit

    Args:
        lab (Array): OKLAB color vector

    Returns:
        Array: SRGB color vector
    """
    # transformation matrices
    M2 = jnp.array(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ]
    )
    M1 = jnp.array(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480],
        ]
    )

    lms_ = jnp.dot(M1, lab)
    lms = lms_**3
    rgb = jnp.dot(M2, lms)

    return rgb


def plot_color_swatches(colors: Array, cols: int = None):
    """Plots color swatches from an array of RGB colors

    Args:
        colors (ArrayLike): Array of RGB colors to plot
        cols (int, optional): Number of columns in plot. Defaults to the number of colors provided.
    """
    colors = colors[jnp.argsort(colors[:, 0])]
    num_colors = colors.shape[0]
    if not cols:
        cols = num_colors
    rows = (num_colors + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    ax = axes.ravel()
    for i, color in enumerate(colors):
        ax[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=np.array(color)))
        ax[i].axis("off")

    # Turn off axes for any remaining empty subplots
    for i in range(num_colors, rows * cols):
        ax[i].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def unform_dist_score(colors: Array, deadzone: float = 0.0) -> float:
    """Score function which takes an array of RGB values as input and outputs a score function
    representing how evenly-spread they are in the OKLAB color space.

    Args:
        colors (Array): Colors to score.
        deadzone (float, optional): Clips the minimum output value at x*global_optima. Defaults to 0..

    Returns:
        float: Score function output. Lower scores represent maximally distant, equidistant colors.
    """

    def dist_sq(v1, v2):
        return jnp.dot(v1 - v2, v1 - v2)

    # remap to OKLAB
    colors = jnp.reshape(colors, (-1, 3))
    ncolors = colors.shape[0]
    colors = vmap(linear_srgb_to_oklab)(colors)

    # distance
    dist_1 = vmap(lambda v1: vmap(lambda v2: dist_sq(v1, v2))(colors))(colors)
    real_dist = dist_1
    triu = jnp.triu_indices(real_dist.shape[0], k=1)
    upper_dists = real_dist[triu]
    upper_dists = jnp.sqrt(
        upper_dists
    )  # sqrt after selecting away diagonal, add small value for stability

    # goal is to minimize, take reciprocoal to heavily penalize low dists
    dist_score = jnp.mean(1 / (upper_dists + 1e-3))  # slight offset to avoid asymptote

    # regression for global optimal score at `ncolors`
    est_min = (
        -0.74862516
        + -0.29700463 * ncolors
        + 0.00310837 * ncolors**2
        + 1.84976861 * ncolors**0.5
    )
    dist_score = jnp.clip(
        dist_score,
        min=est_min * deadzone,
    )
    return dist_score


def gen_equidistant_colors(
    key: Array, n_colors: int = 8, n_samples: int = 1, deadzone: float = 0.0
) -> Array:
    """Randomly generates RGB values for `n_colors` which are optimized to be maximally distant
    and equidistant in the OKLAB color space. Produces `n_samples` random color palettes.

    Args:
        key (Array): Jax RNG key.
        n_colors (int, optional): Number of colors in desired palette. Defaults to 8.
        n_samples (int, optional): Number of distinct random initializations. Defaults to 1.
        deadzone (float, optional): Clips the minimum output value at x*global_optima. Must be >1 for
         unstable `n_colors` values to succeed in optimization, otherwise should be kept 0. Defaults to 0.

    Returns:
        Array: `n_samples` sets of `n_colors` RGB colors. Takes shape (n_samples, n_colors, 3).
    """
    # something is very unstable only when n_colors is 5 or 7, unsure why.
    if (n_colors == 5) or (n_colors == 7):
        assert (
            deadzone >= 1.05
        ), f"`deadzone` >= 1.05 required for `n_colors`={n_colors} "

    # initialize points, boundaries
    buffer = 1e-2  # >0 is necessary for numerical stability of optimization
    mins = buffer * jnp.ones((n_colors * 3,))
    maxs = (1 - buffer) * jnp.ones((n_colors * 3,))
    init = jr.uniform(
        key, shape=(n_samples, n_colors * 3), minval=buffer, maxval=1 - buffer
    )

    # find local optima from initializations
    opto = ProjectedGradient(
        fun=unform_dist_score, projection=projection_box, maxiter=50
    )
    samples = vmap(opto.run, (0, None, None))(init, (mins, maxs), deadzone)
    samples = jnp.reshape(samples.params, (n_samples, -1, 3))  # reshape last axis

    return samples


def posterize_image(
    img: Array,
    n_colors: int = 9,
    output_space: Literal["rgb", "oklab"] = "oklab",
    posterizer: Literal["rgb", "oklab", "none"] = "oklab",
    rand_colors: bool = False,
    key: Array = jr.key(3),
    n_samples: int = 10,
) -> Array:
    """Posterizes provided RGB image using K-means clustering

    Args:
        img (Array): Input image
        n_colors (int, optional): Number of colors to reduce to. Defaults to 9.
        output_space (Literal[&#39;rgb&#39;, &#39;oklab&#39;], optional): Color space for output image. Defaults to 'oklab'.
        posterizer (Literal[&#39;rgb&#39;,&#39;oklab&#39;,&#39;none&#39;], optional): Color space to use when posterizing. Defaults to 'oklab'.
        rand_colors (bool, optional): Whether or not to generate posterization colors at random. Finds maximally distant, equispaced colors. Defaults to False.
        key (Array, optional): Key to use when generating random colors. Defaults to jr.key(3).
        n_samples (int, optional): Number of distinct initializations for random colors. The most uniformly distinct color palette found after `n_samples` initializations is chosen. Defaults to 10.

    Returns:
        Array: Posterized image, color values in `output_space`.
    """
    curr_space = "rgb"  # variable used to track current color space

    # invalid posterizer -> no posterization
    if posterizer not in ["rgb", "oklab"]:
        posterized_img = img
        curr_space = "rgb"
    else:
        # if oklab, map to oklab
        if posterizer == "oklab":
            img = vmap(vmap(linear_srgb_to_oklab))(img)
            curr_space = "oklab"

        # posterize thru k-means clustering of pixels
        arr = img.reshape((-1, 3))
        kmeans = KMeans(n_init="auto", n_clusters=n_colors).fit(arr)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # if rand_colors, switch to random colors
        if rand_colors:
            samples = gen_equidistant_colors(
                key,
                n_colors,
                n_samples=n_samples,
                deadzone=1.05 if (n_samples in [5, 7]) else 0.0,
            )
            scores = vmap(unform_dist_score)(samples)
            best_idx = jnp.unravel_index(jnp.argmax(scores), scores.shape)
            centers = samples[best_idx]
            curr_space = "rgb"

        posterized_img = centers[labels].reshape(img.shape)

    # map output into proper color space
    if (curr_space == "rgb") & (output_space == "oklab"):
        posterized_img = vmap(vmap(linear_srgb_to_oklab))(posterized_img)
        curr_space = "oklab"
    if (curr_space == "oklab") & (output_space == "rgb"):
        posterized_img = vmap(vmap(oklab_to_linear_srgb))(posterized_img)
        curr_space = "rgb"
    return posterized_img


def gaussian_smooth(
    image: Array, kernel_size: int = 9, kernel_std: float = 1.0,
) -> Array:
    """Applies Gaussian smoothing to image. Pads image with edge values before convolving.

    Args:
        image (Array): Image to smooth.
        kernel_size (int, optional): Size of Gaussian kernel. Defaults to 9.
        kernel_std (float, optional): Standard deviation of Gaussian. Defaults to 1..

    Returns:
        Array: _description_
    """
    # smoothing helps prevent pixels from collecting at edges
    x = jnp.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    window = jsp.stats.norm.pdf(x, scale=kernel_std) * jsp.stats.norm.pdf(
        x[:, None], scale=kernel_std
    )
    window = window / window.sum()
    np_cv2 = np.array(window)
    np_cv1 = np.array(image)

    to_stack = []
    for i in range(3):
        smooth_channel = convolve2d(np_cv1[:, :, i], np_cv2, "same", "symm")
        to_stack.append(smooth_channel)
    smooth_image = jnp.stack(to_stack, -1)
    return smooth_image

@jit
def boxcox(lambd, x):
    bc = lax.cond(
            lambd==0.,
            lambda x: jnp.log(x),
            lambda x: (x**lambd - 1)/lambd,
            operand=x,
        )
    return bc

def opt_boxcox(x, box_bounds):
    @jit
    def check_bc_normality(lambd, x):
        bc = boxcox(lambd, x)
        mu = jnp.mean(bc)
        sig = jnp.std(bc)
        logpdfs = jsp.stats.norm.logpdf(bc, mu, sig)
        return -1* jnp.sum(logpdfs)
        
    solver = ProjectedGradient(fun=check_bc_normality, projection=projection_box)
    init = 0.01
    res = solver.run(init, box_bounds, x=x)
    lambd = res.params
    return boxcox(lambd, x), lambd

def inv_truncnorm(x):
    a = -6.5
    b =-0.5
    mu = -3.5
    sig = 1.
    cdf = jsp.stats.norm.cdf
    ppf = jsp.stats.norm.ppf
    return ppf( cdf((a-mu)/sig ) + x * (cdf((b-mu)/sig ) - cdf((a-mu)/sig ) ) )*sig + mu
     
@jit
def blom(x):
    ranks = jsp.stats.rankdata(x, method='average')
    blommed = inv_truncnorm((ranks - 3/8)/(len(ranks)+1/4))
    design = jnp.column_stack([jnp.ones_like(x), x, x**2, x**3, jnp.exp(x), jnp.sin(x), jnp.cos(x)])
    beta = jnp.linalg.inv(design.T @ design) @ design.T @ blommed
    return blommed, beta

class image_sampler:
    def __init__(
        self,
        img: Array,
        num_particles: int = 50000,
        loss_space: Literal["rgb", "oklab"] = "rgb",
        likelihood_params: dict = {"INF":1e2, 'scaled':True},
        posterization_params: dict = {"posterizer": "rgb", "n_colors": 9},
        smoother_params: dict = {"kernel_size": 1, "kernel_std": 1.},
        sampler_params: dict = {
            "annealing_steps": 50,
            "lambd_range": (-1., 2.),
            "extra_steps": 25,
        },
    ):
        """Class to manage image resampling.

        Args:
            img (Array): RGB input image
            num_particles (int, optional): Number of particles in sample. Should be comparable in magnitude to the number of pixels in the image. Defaults to 15000.
            loss_space (Literal[&#39;rgb&#39;, &#39;oklab&#39;], optional): Color space to calculate loss function. Defaults to "rgb".
            likelihood_params (dict, optional): Parameters for likelihood function. Defaults to {"dist_mod": lambda x: (9 * x) ** 2}.
            posterization_params (dict, optional): Parameters for posterization function. Defaults to {"posterizer": "rgb", "n_colors": 9}.
            smoother_params (dict, optional): Parameters for smoothing function. Defaults to {"kernel_size":9, "kernel_std": 1.0}.
            sampler_params (dict, optional): Parameters for sampling function. Defaults to { "annealing_steps": 125, "lambd_range": (-3, -0.5), "extra_steps": 25, }.
        """
        self.img = img
        self.num_particles = num_particles
        self.loss_space = loss_space
        self.xmax = img.shape[0]
        self.ymax = img.shape[1]

        self.likelihood_params = likelihood_params
        self.posterization_params = posterization_params
        self.smoother_params = smoother_params
        self.sampler_params = sampler_params
        self.prior_logpdf = lambda x: 0

    def run(self, key) -> Array:
        """Samples movement of pixels over time, forming an approximation of the original image.

        Posterizes image -> initializes particle samples from posterized palette -> creates smoothed
        reference image for loss function -> creates loss function -> samples particle movement.

        Returns:
            Array: (T, num_particles, 5) array. Last axis stores [x,y,R,G,B] for each particle for each
            timestep.
        """
        # items stored as class objects to make their retreival easier
        key1, key2, key3 = jr.split(key, 3)
        # generate posterized palette
        self.palette = posterize_image(
            img=self.img,
            key=key1,
            output_space=self.loss_space,
            **self.posterization_params,
        )
        z_init = self._initialize_samps(
            self.num_particles, key2
        )  # initialize particles
        # create reference image for loss function
        self.ref_img = gaussian_smooth(image=self.palette, **self.smoother_params)
        # create loss function from reference image
        self._log_likelihood = self._gen_likelihood_function(**self.likelihood_params)
        self.samples_out = self._sample_tempered_hmc(z_init, key3)  
        frames = self.samples_out[0]  # samples particle movement
        # convert to RGB if working in OKLAB space
        if self.loss_space.lower() == "oklab":
            frames = frames.at[:, :, 2:].set(
                vmap(vmap(oklab_to_linear_srgb))(frames[:, :, 2:])
            )
        # clip for safety
        frames = frames.at[:, :, 2:].set(jnp.clip(frames[:, :, 2:], 0, 1))
        self.frames = frames
        return frames

    def _initialize_samps(self, n_particles, key):
        key1, key2 = jr.split(key, 2)
        max_vals = jnp.array([self.xmax, self.ymax], dtype=jnp.int32)
        coords_init = jr.randint(key1, (n_particles, 2), minval=0, maxval=max_vals)

        pixel_bucket = jnp.reshape(self.palette, (-1, 3))
        unique_colors, counts = jnp.unique(pixel_bucket, axis=0, return_counts=True)

        # white is blank space when plotting, so omit from color space
        if self.loss_space.lower() == "rgb":
            valid = jnp.any(unique_colors < 0.92, axis=-1)
        else:
            valid = jnp.all(
                jnp.abs(unique_colors - jnp.array([1, 0, 0])) > 0.001, axis=-1
            )

        # instead omit modal color in case of random colormap
        if self.posterization_params.get("posterizer") == "rand":
            mode = unique_colors[jnp.argmax(counts)]
            valid = jnp.all(unique_colors != mode, axis=1)
        unique_colors, counts = unique_colors[valid], counts[valid]

        probabilities = counts / jnp.sum(counts)
        sampled_indices = jr.choice(
            key2, len(unique_colors), shape=(n_particles,), p=probabilities
        )
        colors = unique_colors[sampled_indices]

        z_init = jnp.concat([coords_init, colors], axis=-1)
        return z_init

    def _gen_likelihood_function(self, INF=1e2, scaled=True):
        @jit
        def log_likelihood(z):
            def distance(x, y):
                return jnp.sqrt(jnp.sum((x - y) ** 2))
            coords = z[:2]
            colors = z[2:]
            floor_z = jnp.floor(coords)
            floor_z = jnp.astype(floor_z, jnp.int32)

            # ensure pixel coords are inside the image
            out_of_bounds = (
                (floor_z[0] < 0)
                | (floor_z[0] > self.xmax - 1)
                | (floor_z[1] < 0)
                | (floor_z[1] > self.ymax - 1)
            )
            value = lax.cond(
                out_of_bounds,
                lambda *_: -INF,
                lambda arg: -1
                * distance(self.ref_img[arg[0], arg[1]], colors),
                operand=floor_z,
            )
            return value
        
        if scaled:
            loss_maps, (uq_colors, invs) = self._gen_loss_maps(log_likelihood)
            uq_losses = jnp.unique(loss_maps.reshape((loss_maps.shape[0], -1)), axis=1)
            blommed, betas = vmap(blom)(uq_losses)
            sigs = jnp.std(blommed, -1)

            @jit
            def log_likelihood(z):
                def distance(x, y):
                    return jnp.sqrt(jnp.sum((x - y) ** 2))
                def find_index(value):
                    match = vmap(lambda x: distance(x,value))(uq_colors)#jnp.all(uq_colors == value, axis=1)
                    return jnp.argmin(match)
                coords = z[:2]
                colors = z[2:]
                floor_z = jnp.floor(coords)
                floor_z = jnp.astype(floor_z, jnp.int32)

                # ensure pixel coords are inside the image
                out_of_bounds = (
                    (floor_z[0] < 0)
                    | (floor_z[0] > self.xmax - 1)
                    | (floor_z[1] < 0)
                    | (floor_z[1] > self.ymax - 1)
                )
                idx = find_index(colors)
                dist = -1*distance(self.ref_img[floor_z[0], floor_z[1]], colors)
                design = jnp.column_stack([jnp.ones_like(dist), dist, dist**2, dist**3, jnp.exp(dist), jnp.sin(dist), jnp.cos(dist)]).squeeze()
                blommed_dist = design @ betas[idx]
                value = lax.cond(
                    out_of_bounds,
                    lambda *_: -INF,
                    lambda *_: blommed_dist, #/ sigs[idx],
                    operand=floor_z,
                )
                return value
        return log_likelihood

    def _sample_tempered_hmc(self, init, key):
        samp_params = self.sampler_params.copy()
        n_samples = self.num_particles
        annealing_steps = samp_params.pop("annealing_steps", 125)
        lambd_range = samp_params.pop("lambd_range", (-1., 2.))
        extra_steps = samp_params.pop("extra_steps", 25)

        key, subkey = jr.split(key)

        def railed_smc_inference_loop(
            rng_key, smc_kernel, initial_state, lambda_schedule
        ):
            """Run the temepered SMC algorithm."""

            def one_step(carry, lambd):
                i, state, key = carry
                key = jr.fold_in(key, i)
                state, _ = smc_kernel(key, state, lambd)
                new_particles = state.particles
                new_particles = new_particles.at[:, 2:].set(init[:, 2:])
                state = TemperedSMCState(new_particles, state.weights, state.lmbda)
                return (i + 1, state, key), state

            n_iter, final_state = lax.scan(
                one_step, (0, initial_state, rng_key), lambda_schedule
            )

            return n_iter, final_state

        inv_mass_matrix = jnp.diag(
            jnp.array([2, 2, 1e-9, 1e-9, 1e-9])
        )  # really small values effectively freeze momentum
        hmc_parameters = dict(
            step_size=1., inverse_mass_matrix=inv_mass_matrix, num_integration_steps=1
        )

        schedule_tempered = blackjax.tempered_smc(
            self.prior_logpdf,
            self._log_likelihood,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            extend_params(n_samples, hmc_parameters),
            resampling.systematic,
            num_mcmc_steps=1,
        )

        initial_smc_state = schedule_tempered.init(init)
        schedule = jnp.logspace(lambd_range[0], lambd_range[1], annealing_steps)
        schedule = jnp.pad(
            schedule,
            [(0, extra_steps)],
            mode="constant",
            constant_values=(schedule[-1],),
        )
        _, smc_samples = railed_smc_inference_loop(
            subkey, schedule_tempered.step, initial_smc_state, schedule
        )
        return smc_samples

    def _gen_loss_maps(self, log_likelihood_fn):
        # get unique color values
        colors = self.palette
        colors = jnp.reshape(colors, (-1, 3))
        colors, invs = jnp.unique(colors, axis=0, return_inverse=True)

        # get coordinate array to calculate loss values at, same shape as img
        x = jnp.arange(self.xmax)
        y = jnp.arange(self.ymax)
        xx, yy = jnp.meshgrid(x, y, indexing="ij")
        coordinates_array = jnp.stack((xx, yy), axis=-1)

        # generate images
        def color2map(color):
            values = jnp.array(color).reshape(1, 1, 3)
            uniform_img = jnp.tile(values, (self.xmax, self.ymax, 1))
            loss_ready = jnp.concat([coordinates_array, uniform_img], axis=-1)
            loss_map = vmap(vmap(log_likelihood_fn))(loss_ready)
            return loss_map
        loss_maps = vmap(color2map)(colors)
        return loss_maps, (colors, invs)

    def show_loss_map(self):
        """Prints a heatmap of the negative loglikelihood function for each color. Must be run after `run`. Useful for debugging."""
        # get unique color values
        loss_maps, (_,_) = self._gen_loss_maps(self._log_likelihood)

        nrowsfn = lambda x: x // 4 + 1 if x % 4 != 0 else x // 4  # noqa: E731
        fig, axes = plt.subplots(nrowsfn(len(loss_maps)), 4, sharex=True, sharey=True)
        ax = axes.ravel()

        # plot
        for i, loss_map in enumerate(loss_maps):
            pic = ax[i].imshow(-1*loss_map, cmap="grey", aspect="auto")
        fig.colorbar(pic, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
        means = jnp.array([jnp.mean(m) for m in loss_maps])
        stds = jnp.array([jnp.std(m) for m in loss_maps])
        mins = jnp.array([jnp.min(m) for m in loss_maps])
        maxs = jnp.array([jnp.max(m) for m in loss_maps])
        print(f'means:\n{means}\nstds:\n{stds}')
        print()
        print(f'mins:\n{mins}\nmaxs:\n{maxs}')

        plt.show()
    def _reshape_frames(self, frames:Array, smoothing_params:dict=None)->Array:
        """Reorganizes random particle samples into a sequence of images.

        Args:
            frames (Array): Sample frames to reshape into images.
            smoothing_params (dict, optional): Smoothing parameters to apply to each image. Defaults to None.

        Returns:
            Array: Array of images.
        """
        n, _, _ = frames.shape
        output_array = jnp.zeros((n, self.xmax, self.ymax, 3))
        
        # Iterate over the batch dimension
        for i in range(n):
            indices = frames[i, :, :2].astype(int)
            values = frames[i, :, 2:]
            output_array = output_array.at[i, indices[:, 0], indices[:, 1]].add(values)
            index_counts = jnp.zeros_like(output_array[i])
            index_counts = index_counts.at[indices[:, 0], indices[:, 1]].add(1.)
            whites = jnp.all(index_counts == 0, axis=-1)
            index_counts = jnp.clip(index_counts, 1.,)
            output_array = output_array.at[i, :, :].set(output_array[i] / index_counts)
            output_array = output_array.at[i].add(whites[:,:,None].astype(jnp.float32))
            if smoothing_params:
                in_oklab = vmap(vmap(linear_srgb_to_oklab))(output_array[i])
                smoothed_oklab = gaussian_smooth(in_oklab, **smoothing_params)
                smoothed_rgb = jnp.clip (vmap(vmap(oklab_to_linear_srgb))(smoothed_oklab) , 0., 1.)
                whered_image = jnp.where(whites[:,:, None], smoothed_rgb, output_array[i])
                output_array = output_array.at[i].set(whered_image)
        return output_array

    def draw_gif(self, gif_loc: str = None, start_frame:int=0,interval: int = 50, render:Literal['scatter', 'pixel']='pixel', smoothing_params:dict=None) -> FuncAnimation:
        """Creates GIF from samples stored in class object. Must be run after `self.run`.

        Args:
            gif_loc (str, optional): File location to save GIF. Does not save if None. Defaults to None.
            start_frame (int, optional): Sample step to begin GIF on. Defaults to 0.
            interval (int, optional): Delay in ms between frames. Defaults to 50.
            render (Literal[&#39;scatter&#39;, &#39;img&#39;], optional): Method to render images. Defaults to 'pixel'.
            smoothing_params (dict, optional): `gaussian_filter` params to apply to frames of GIF if `render='pixel'`. Defaults to None.
        
        Returns:
            FuncAnimation: Matplotlib FuncAnimation object
        """
        
        fig, ax = plt.subplots()
        # initialize plot with noisy data
        if render=='scatter':
            scatter = ax.scatter(self.frames[0][:, 1], self.frames[0][:, 0])
            frames = self.frames[start_frame:]

        else:
            img = ax.imshow(np.ones_like(self.img))
            frames = self._reshape_frames(self.frames[start_frame:], smoothing_params)

        # params dump to prevent axes, any whitespace
        # some are probabily un-necessary, but it works
        fig.set_size_inches((6 * self.ymax / self.xmax, 6), forward=True)
        ax.axis("off")
        ax.margins(0)
        fig.tight_layout(pad=0.0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(0, self.ymax)  # x and y are swapped for some reason
        ax.set_ylim(0, self.xmax)

        def update(frame_data):
            if render == 'scatter':
                offsets = np.column_stack(
                    (frame_data[:, 1], self.xmax - frame_data[:, 0])
                )  # it otherwise prints upside down
                colors = frame_data[:, 2:]
                scatter.set_offsets(offsets)
                scatter.set_color(np.array(colors))
                scatter.set_sizes(np.pi * np.ones(frame_data.shape[0]))
                return (scatter,)
            else:
                img.set_data(jnp.flip(frame_data, axis=0))
                return (img,)

        ani = FuncAnimation(
            fig,
            update,
            frames=frames,
            blit=True,
            interval=interval,
        )

        # Save the animation
        if gif_loc:
            writer = animation.PillowWriter(fps=20, bitrate=1800)
            ani.save(
                gif_loc,
                writer=writer,
                savefig_kwargs={
                    "pad_inches": 0,
                },
            )
        return ani
