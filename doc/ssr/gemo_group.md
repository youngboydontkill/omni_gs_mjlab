  MJLab 里的 geom group 在这个仓库中主要被当作“几何类别过滤标签”使用，尤其影响 raycast / camera 这类传感器看到什么；它不是物理碰撞是否发生的主开关。真正控制 MuJoCo 物理碰撞的是 contype / conaffinity。
    结合 Kuavo 当前配置，可以按这张表理解：

    Kuavo S54 的关键例子在 src/omni_gs_playground/assets/robots/kuavo/biped_s54/xml/biped_s54.xml:11：
  
    <default class="collision">
      <geom material="collision" contype="4" conaffinity="4" user="2" group="3" .../>
      <default class="collision-left">
        <geom contype="1" conaffinity="2" group="3" .../>
      </default>
      <default class="collision-right">
        <geom contype="2" conaffinity="1" group="3" .../>
      </default>
    </default>
    这里有两个层次：
  1. group="3"：把这些 geom 标成 collision 显示/过滤组。
  2. contype / conaffinity：真正决定哪些 geom 之间产生 contact。
    比如 S54 左脚碰撞体属于 collision-left，右脚属于 collision-right。左脚 contype=1, conaffinity=2，右脚 contype=2, conaffinity=1，这类设置可以让左右侧之间按位匹配产生碰撞/接触关系。躯干默认 collision 则用 contype=4, conaffinity=4。这和group=3 是两套机制。
    
    S54 的 visual mesh 则基本都是 group="1" 且：
    contype="0" conaffinity="0"
    例如 torso、leg、foot、arm 的 STL mesh 都是这样。这意味着它们不参与物理碰撞，但仍然可能被 raycast 这类几何查询命中。如果raycast 默认包含 (0,1,2)，它会把 group 1 的机器人外观 mesh 当成可命中的几何体。
    
    这正是前面 S54 SSR foothold planning scanner 的问题根因：ssr_foothold_planning_scan 原先没有限制 terrain group，只排除了挂载的base_link 父 body。exclude_parent_body=True 只排除传感器挂载的 parent body，不会排除整台机器人。因此从 base 向下打射线时，仍可
    能命中腿部/脚部的 group-1 visual mesh，导致 planning map 高度异常，进而让 foothold support reward 近似恒为 0。
    
    现在 S54 SSR 的 planning scanner 已改成只扫 terrain group 0
    foothold_planning_scan = RayCastSensorCfg(
    name="ssr_foothold_planning_scan",
    ...
    exclude_parent_body=True,
    include_geom_groups=(0,),
    )
    
    S45-Rough 的 terrain_scan 也是同样思路
    terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    ...
    include_geom_groups=(0,),
    )
    
    这个配置的含义是：高度图、落足规划这类“只想知道地形”的 raycast，必须显式限制 include_geom_groups=(0,)。否则 raycast 看到机器人visual mesh 是合理行为，不是 MuJoCo 碰撞配置错误。
    depth camera 的策略不同。Kuavo S54 默认 depth camera 在 src/omni_gs_playground/tasks/velocity/velocity_env_cfg.py:700 开启了：
    enabled_geom_groups=(0, 1, 2, 3, 4, 5)
    这表示相机渲染时可以看到地形、机器人 visual、collision/debug 等多个组。对相机来说这通常合理，因为机器人自遮挡、可视模型、场景都可能是渲染语义的一部分。但对 foothold planning / terrain height scan 不合理，因为它们需要的是“地形高度”，不是“最近几何体高度”。
    所以总结成工程规则：
  - contype/conaffinity：决定物理接触是否产生。
  - group：决定几何体属于哪个显示/查询组。
  - camera 可以根据需要打开多个 group。
  - terrain raycast / foothold planning raycast 应该只 include group 0。
  - exclude_parent_body=True 不是“排除机器人自身”的完整机制，只能排除挂载 body，不能替代 include_geom_groups=(0,)。
  - Kuavo 的 visual mesh 是 group 1、无碰撞；collision primitive 是 group 3、有 contype/conaffinity；terrain 是 group 0。
  - 如果 raycast 用默认组扫，会出现“无物理碰撞的 visual mesh 仍被传感器命中”的问题，这就是 S54 SSR planning map 之前异常的核心原因。