"""AME-Locomotion training terrains, ported to the mjlab terrain interface.

Ported from ``AME_Locomotion/source/.../ame_locomotion/terrains/``:

- ``HfConcentricGapTerrainCfg``: the custom concentric gap/ground-ring
  heightfield (``loco_hf_terrains.concentric_gap_terrain``). This is the only
  sub-terrain of the AME training set with no mjlab equivalent.
- ``AME_ROUGH_TERRAINS_CFG``: mirrors the AME ``terrain_cfg.ROUGH_TERRAINS_CFG``
  (size 8x8, border 50, 10 rows, 8 sub-terrains with the source proportions),
  mapping each Isaac Lab terrain type to its mjlab counterpart.

mjlab ships as a PyPI dependency, so the custom heightfield lives here in the
project instead of in mjlab itself. The heightfield follows the mjlab
``SubTerrainCfg.function`` contract (see ``HfDiscreteObstaclesTerrainCfg`` for
the negative-obstacle offset convention).
"""

import uuid
from dataclasses import dataclass

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)


@dataclass(kw_only=True)
class HfConcentricGapTerrainCfg(SubTerrainCfg):
  """Concentric alternating gap/ground rings with a flat central platform.

  Ported faithfully from AME's ``concentric_gap_terrain``: the tile is filled
  with concentric rings starting from a deep gap at the outer edge, alternating
  with ground bands whose height is sampled in
  ``[-ground_height_max, ground_height_max]``, and ending in a flat central
  platform at ground level. Gaps drop to ``gap_depth`` below the reference
  ground (z=0), so the elevation-map sees them as deep holes (z ≈ -1.2 after
  clamping).
  """

  gap_width_range: tuple[float, float]
  """Min/max width of the gap rings (m), interpolated by difficulty."""
  ground_width_range: tuple[float, float]
  """Min/max width of the ground rings (m); wider at low difficulty."""
  ground_height_max: float
  """Max absolute ground-band height above/below the reference ground (m)."""
  gap_depth: float = -2.0
  """Depth of the gaps (negative obstacles), in meters."""
  platform_width: float = 1.0
  """Side length of the flat central platform (m)."""
  border_width: float = 0.0
  """Kept for parity with the AME config; the source algorithm ignores it."""
  horizontal_scale: float = 0.1
  """Heightfield grid resolution along x and y, in meters per cell.

  AME uses 0.05 m; mjlab terrain defaults to 0.1 m. The finer 0.05 m grid can
  overflow MuJoCo Warp's 50-contact-point-per-pair hfield limit (see
  ``AME_ROUGH_TERRAINS_CFG``), so the training terrain runs at 0.1 m.
  """
  vertical_scale: float = 0.005
  """Heightfield height resolution, in meters per integer unit of the noise array."""
  base_thickness_ratio: float = 1.0
  """Ratio of the heightfield base thickness to its maximum surface height."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    """Generate the concentric gap heightfield (source algorithm, mjlab plumbing)."""
    body = spec.body("terrain")

    gap_depth_px = int(abs(self.gap_depth) / self.vertical_scale)
    gap_width = int(
      round(
        (self.gap_width_range[0] + difficulty * (self.gap_width_range[1] - self.gap_width_range[0]))
        / self.horizontal_scale
      )
    )
    ground_width = int(
      round(
        (self.ground_width_range[0] + (1.0 - difficulty) * (self.ground_width_range[1] - self.ground_width_range[0]))
        / self.horizontal_scale
      )
    )
    ground_height_max_px = int(round(self.ground_height_max / self.vertical_scale))
    width_pixels = int(self.size[0] / self.horizontal_scale)
    length_pixels = int(self.size[1] / self.horizontal_scale)
    platform_px = int(self.platform_width / self.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    is_gap = True
    while (stop_x - start_x) > platform_px and (stop_y - start_y) > platform_px:
      if is_gap:
        hf_raw[start_x:stop_x, start_y:stop_y] = -gap_depth_px
        start_x += gap_width
        stop_x -= gap_width
        start_y += gap_width
        stop_y -= gap_width
      else:
        h = int(rng.integers(-ground_height_max_px, ground_height_max_px + 1))
        hf_raw[start_x:stop_x, start_y:stop_y] = h
        start_x += ground_width
        stop_x -= ground_width
        start_y += ground_width
        stop_y -= ground_width
      is_gap = not is_gap

    # Flat central platform at ground level (z = 0).
    x1 = (width_pixels - platform_px) // 2
    x2 = (width_pixels + platform_px) // 2
    y1 = (length_pixels - platform_px) // 2
    y2 = (length_pixels + platform_px) // 2
    hf_raw[x1:x2, y1:y2] = 0

    noise = np.rint(hf_raw).astype(np.int16)

    elevation_min = np.min(noise)
    elevation_max = np.max(noise)
    elevation_range = (
      elevation_max - elevation_min if elevation_max != elevation_min else 1
    )
    max_physical_height = elevation_range * self.vertical_scale
    base_thickness = max_physical_height * self.base_thickness_ratio
    normalized_elevation = (noise - elevation_min) / elevation_range

    unique_id = uuid.uuid4().hex
    field = spec.add_hfield(
      name=f"hfield_{unique_id}",
      size=[
        self.size[0] / 2,
        self.size[1] / 2,
        max_physical_height,
        base_thickness,
      ],
      nrow=noise.shape[0],
      ncol=noise.shape[1],
      userdata=normalized_elevation.flatten().astype(np.float32).tolist(),
    )

    # Offset the hfield down by the deepest elevation so that noise=0 (ground
    # rings and platform) sits at world z=0 — same convention as mjlab's
    # HfDiscreteObstaclesTerrainCfg "choice" mode.
    hfield_z_offset = elevation_min * self.vertical_scale
    material_name = color_by_height(spec, noise, unique_id, normalized_elevation)

    hfield_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=[self.size[0] / 2, self.size[1] / 2, hfield_z_offset],
      material=material_name,
    )

    origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
    geom = TerrainGeometry(geom=hfield_geom, hfield=field)
    return TerrainOutput(origin=origin, geometries=[geom])


# AME ``ROUGH_TERRAINS_CFG`` (training), mapped onto mjlab terrain types.
# Proportions and per-terrain parameters are copied from the source
# ``terrain_cfg.py``. The heightfield terrains run at ``horizontal_scale=0.1``
# (mjlab default) instead of the source's 0.05: the finer grid overflows MuJoCo
# Warp's per-pair hfield contact limit (50 cells) when a flat body rests on it.
AME_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=50.0,
  num_rows=10,
  num_cols=20,
  difficulty_range=(0.0, 1.0),
  sub_terrains={
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.1,
      step_height_range=(0.05, 0.2),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
      holes=False,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.1,
      step_height_range=(0.05, 0.2),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
      holes=False,
    ),
    "boxes": terrain_gen.BoxRandomGridTerrainCfg(
      proportion=0.1,
      grid_width=0.45,
      grid_height_range=(0.05, 0.2),
      platform_width=2.0,
    ),
    # horizontal_scale 0.05 (AME source fidelity) is too fine for MuJoCo Warp's
    # hfield collision, which caps contact points per geom pair at 50: a flat
    # body on a 0.05-m grid covers >50 cells. 0.1 m (mjlab default) keeps a
    # fallen torso at ~15 cells. The elevation-map observation grid (0.05 m)
    # is independent and unchanged.
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.1,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      downsampled_scale=0.1,
      border_width=0.25,
      horizontal_scale=0.1,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 0.4),
      platform_width=2.0,
      border_width=0.25,
      horizontal_scale=0.1,
    ),
    "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 0.4),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
      horizontal_scale=0.1,
    ),
    # AME uses Isaac Lab's HF stepping stones (near-flat ±0.05 m pads over a
    # 2 m pit). mjlab's box stepping stones approximate the same layout with
    # stone tops at z = stone_height above the reference ground and a floor at
    # ``floor_depth`` below it.
    "hf_steppingstones": terrain_gen.BoxSteppingStonesTerrainCfg(
      proportion=0.2,
      stone_size_range=(0.25, 0.5),
      stone_distance_range=(0.05, 0.25),
      stone_height=0.05,
      stone_height_variation=0.05,
      stone_size_variation=0.1,
      floor_depth=2.0,
      platform_width=2.0,
      border_width=0.25,
    ),
    "hf_gaps": HfConcentricGapTerrainCfg(
      proportion=0.2,
      gap_width_range=(0.1, 0.5),
      ground_width_range=(0.5, 0.5),
      ground_height_max=0.025,
      gap_depth=-2.0,
      platform_width=2.0,
      border_width=0.25,
    ),
  },
  add_lights=True,
)
