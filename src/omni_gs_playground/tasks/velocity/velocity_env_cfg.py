"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
import mjlab.terrains as terrain_gen
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import flat, ROUGH_TERRAINS_CFG
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from omni_gs_playground.tasks.velocity import mdp

NAVIGATION_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=10,
  difficulty_range=(0.0, 1.0),
  sub_terrains={
    "flat": flat(proportion=0.3),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=(0.02, 0.14),
      step_width=0.32,
      platform_width=2.0,
      border_width=0.8,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.02, 0.14),
      step_width=0.32,
      platform_width=2.0,
      border_width=0.8,
    ),
    # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
    #   proportion=0.1,
    #   slope_range=(0.0, 0.5),
    #   platform_width=2.0,
    #   border_width=0.25,
    #   inverted=True,
    # ),
    "tilted_grid": terrain_gen.BoxTiltedGridTerrainCfg(
      proportion=0.1,
      grid_width=0.75,
      tilt_range_deg=24.0,
      height_range=0.35,
      platform_width=1.2,
      border_width=0.25,
      floor_depth=1.5,
    ),
    "box_random_grid": terrain_gen.BoxRandomGridTerrainCfg(
      proportion=0.10,
      grid_width=0.45,
      grid_height_range=(0.02, 0.45),
      platform_width=1.2,
      border_width=0.25,
      merge_similar_heights=True,
    ),
    "nested_rings": terrain_gen.BoxNestedRingsTerrainCfg(
      proportion=0.05,
      num_rings=8,
      ring_width_range=(0.15, 0.45),
      gap_range=(0.1, 0.2),
      height_range=(0.0, 0.10),
      platform_width=1.2,
      border_width=0.25,
      floor_depth=1.5,
    ),
    "narrow_beams": terrain_gen.BoxNarrowBeamsTerrainCfg(
      proportion=0.05,
      num_beams=12,
      beam_width_range=(0.18, 0.45),
      beam_height=0.25,
      spacing=0.7,
      platform_width=1.2,
      border_width=0.25,
      floor_depth=1.5,
    ),
    "stepping_stones": terrain_gen.BoxSteppingStonesTerrainCfg(
      proportion=0.05,
      stone_size_range=(0.35, 0.75),
      stone_distance_range=(0.15, 0.45),
      stone_height=0.25,
      stone_height_variation=0.18,
      stone_size_variation=0.18,
      displacement_range=0.12,
      platform_width=1.2,
      border_width=0.25,
      floor_depth=1.5,
    ),
    # "random_spread_boxes": terrain_gen.BoxRandomSpreadTerrainCfg(
    #   proportion=0.05,
    #   num_boxes=60,
    #   box_width_range=(0.2, 1.2),
    #   box_length_range=(0.2, 1.2),
    #   box_height_range=(0.1, 2.0),
    #   platform_width=1.0,
    #   border_width=0.25,
    #   add_floor=True,
    # ),
    # "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
    #   proportion=0.10,
    #   obstacle_width_range=(0.25, 1.0),
    #   obstacle_height_range=(0.1, 1.0),
    #   num_obstacles=60,
    #   platform_width=1.0,
    #   border_width=0.25,
    # ),
  },
  add_lights=True,
)

GEOLOCO_TERRAINS_CFG = TerrainGeneratorCfg(
    # curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        # ============================================================
        # Up stairs: 25%
        # GeoLoco: MeshPyramidStairs + HfSteppingStones as noisy stairs
        # mjlab: BoxPyramidStairs + BoxRandomStairs approximation
        # ============================================================
        "stairs_up_26": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_30": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_34": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_noisy": terrain_gen.BoxRandomStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.0, 0.22),
            step_width=0.30,
            platform_width=2.5,
            border_width=1.0,
        ),

        # ============================================================
        # Down stairs: 25%
        # GeoLoco: MeshInvertedPyramidStairs + slope/discrete variants
        # ============================================================
        "stairs_down_26": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_30": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_34": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.15, 0.35),
            platform_width=2.0,
            border_width=0.25,
            inverted=True,
        ),
        "stairs_down_discrete": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.05,
            grid_width=0.30,
            grid_height_range=(0.0, 0.18),
            platform_width=2.5,
            border_width=0.25,
            merge_similar_heights=True,
        ),

        # ============================================================
        # Crossing terrains: 25%
        # GeoLoco: MeshGap + MeshPit
        # mjlab: use stepping stones / nested rings / random grid as
        # approximation for gap, pit and foot-placement terrain.
        # ============================================================
        "gap_easy": terrain_gen.BoxSteppingStonesTerrainCfg(
            proportion=0.06,
            stone_size_range=(0.55, 0.85),
            stone_distance_range=(0.05, 0.20),
            stone_height=0.18,
            stone_height_variation=0.08,
            stone_size_variation=0.10,
            displacement_range=0.05,
            platform_width=2.5,
            border_width=0.25,
            floor_depth=1.0,
        ),
        "gap_medium": terrain_gen.BoxSteppingStonesTerrainCfg(
            proportion=0.05,
            stone_size_range=(0.45, 0.75),
            stone_distance_range=(0.10, 0.35),
            stone_height=0.20,
            stone_height_variation=0.12,
            stone_size_variation=0.15,
            displacement_range=0.08,
            platform_width=2.2,
            border_width=0.25,
            floor_depth=1.2,
        ),
        # "gap_hard": terrain_gen.BoxSteppingStonesTerrainCfg(
        #     proportion=0.04,
        #     stone_size_range=(0.35, 0.65),
        #     stone_distance_range=(0.25, 0.50),
        #     stone_height=0.22,
        #     stone_height_variation=0.15,
        #     stone_size_variation=0.18,
        #     displacement_range=0.12,
        #     platform_width=1.8,
        #     border_width=0.25,
        #     floor_depth=1.5,
        # ),
        # "pit": terrain_gen.BoxNestedRingsTerrainCfg(
        #     proportion=0.05,
        #     num_rings=5,
        #     ring_width_range=(0.30, 0.60),
        #     gap_range=(0.05, 0.25),
        #     height_range=(0.05, 0.30),
        #     platform_width=2.0,
        #     border_width=0.25,
        #     floor_depth=0.35,
        # ),
        # "pit_double": terrain_gen.BoxNestedRingsTerrainCfg(
        #     proportion=0.05,
        #     num_rings=8,
        #     ring_width_range=(0.25, 0.55),
        #     gap_range=(0.10, 0.35),
        #     height_range=(0.05, 0.25),
        #     platform_width=1.5,
        #     border_width=0.25,
        #     floor_depth=0.45,
        # ),

        # ============================================================
        # Support terrains: 25%
        # Rough / slope / boxes for visual sim2real generalization
        # ============================================================
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(-0.03, 0.06),
            noise_step=0.02,
            border_width=0.25,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.25,
            inverted=False,
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.25,
            inverted=True,
        ),
        "boxes": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.05,
            grid_width=0.40,
            grid_height_range=(0.0, 0.15),
            platform_width=2.0,
            border_width=0.25,
            merge_similar_heights=True,
        ),
    },
    add_lights=True,
)


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  ##
  # Observations
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / terrain_scan.max_distance,
    ),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.05,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 2.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      }
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
          {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace( GEOLOCO_TERRAINS_CFG,# GEOLOCO_TERRAINS_CFGNAVIGATION_TERRAINS_CFG
                                  curriculum=True,),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )


def make_kuavo_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the shared Kuavo S54 velocity tracking task configuration."""
  cfg = make_velocity_env_cfg()

  depth_camera = CameraSensorCfg(
    name="depth",
    # parent_body="robot/zhead_2_link",
    # pos=(0.093839,0.0475,0.050077),
    # mujoco camera的optical frame是-z, 神奇的定义
    # quat=(0.577506, 0.408028, -0.408028, -0.577506),
    parent_body="robot/waist_yaw",
    pos=(0.0787, 0.0, -0.028449),
    quat=(0.624338, 0.331967, -0.331967, -0.624338),
    fovy=68.0, # for orbbec gemini 335l
    # fovy=58.0, # for d435i
    width=42,
    height=42,
    data_types=("depth",),
    enabled_geom_groups=(0, 1, 2, 3, 4, 5),
    use_textures=False,
    use_shadows=False,
  )

  cfg.scene.sensors = (depth_camera,)

  for group in cfg.observations.values():
    group.terms.pop("phase", None)
    group.terms.pop("height_scan", None)
    group.terms["depth"] = ObservationTermCfg(
      func=mdp.depth_image_obs,
      params={"sensor_name": depth_camera.name, "flatten": True},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    )

  cfg.observations["actor"].terms["base_ang_vel"].params["sensor_name"] = (
    "robot/BodyGyro"
  )
  cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = (
    "robot/BodyVel"
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.ranges.lin_vel_x = (0.0, 1.0)  # only forward command for depth 

  cfg.rewards.pop("foot_gait", None)
  cfg.rewards.pop("angular_momentum", None)

  cfg.curriculum.pop("command_vel", None)  # 不需要速度课程
  return cfg
