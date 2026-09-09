"""GONG transfer for synthetic filament transmission."""

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import StaticConfig


def block_average(image: np.ndarray, factor: int) -> np.ndarray:
    """Average the last two axes in factor-by-factor blocks (crop remainder)."""
    if factor <= 1:
        return image
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    ny, nx = image.shape[-2:]
    ny_c, nx_c = (ny // factor) * factor, (nx // factor) * factor
    cropped = image[..., :ny_c, :nx_c]
    return cropped.reshape(
        *image.shape[:-2],
        ny_c // factor,
        factor,
        nx_c // factor,
        factor,
    ).mean(axis=(-3, -1))


def degrade_transmission(tau_map: np.ndarray, config: StaticConfig) -> np.ndarray:
    """Form native GONG transmission from oversampled ``exp(-tau)``. No noise.

    Compositing path for real backgrounds: only the filament's transmission
    field receives the synthetic PSF; the crop it lands on keeps its genuine
    PSF and noise. Transmission is linear in the emergent intensity
    (I = I_bg*T + S*(1-T)), so T is the correct domain to blur.
    """
    transmission = np.exp(-tau_map)
    blurred = gaussian_filter(transmission, sigma=config["psf_sigma_px"])
    downsampled = block_average(blurred, config["downsample_factor"])
    return np.clip(downsampled, 0.0, 1.0)


def degrade_transmission_batch(
    tau_maps: np.ndarray,
    config: StaticConfig,
) -> np.ndarray:
    """Form native transmission for a batch of high-resolution tau maps."""
    tau_maps = np.asarray(tau_maps, dtype=float)
    expected_shape = (config["ny"], config["nx"])
    if tau_maps.ndim != 3 or tau_maps.shape[1:] != expected_shape:
        raise ValueError(
            "tau_maps must have shape "
            f"(batch, {config['ny']}, {config['nx']}); received {tau_maps.shape}"
        )
    if tau_maps.shape[0] == 0:
        return np.empty(
            (
                0,
                config["ny"] // config["downsample_factor"],
                config["nx"] // config["downsample_factor"],
            ),
            dtype=float,
        )
    transmission = np.exp(-tau_maps)
    blurred = gaussian_filter(
        transmission,
        sigma=(0.0, config["psf_sigma_px"], config["psf_sigma_px"]),
    )
    factor = config["downsample_factor"]
    if factor <= 1:
        return np.clip(blurred, 0.0, 1.0)
    batch, ny, nx = blurred.shape
    ny_c, nx_c = (ny // factor) * factor, (nx // factor) * factor
    cropped = blurred[:, :ny_c, :nx_c]
    downsampled = cropped.reshape(
        batch,
        ny_c // factor,
        factor,
        nx_c // factor,
        factor,
    ).mean(axis=(2, 4))
    return np.clip(downsampled, 0.0, 1.0)
