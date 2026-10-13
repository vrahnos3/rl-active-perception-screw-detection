import numpy as np


def clear_camera_rays(env) -> None:
    """Remove the preceding detector query's ray overlay, if present."""
    node = env.supervisor.getFromDef('DEBUG_CAMERA_RAYS')
    if node is not None:
        node.remove()


def draw_camera_rays(env, rays) -> None:
    """Draw the latest query's world-space rays, invisible to the robot camera.

    Each item contains origin, the normalized v4 direction and intersection.
    Cyan: forward ray and arrow. Magenta: opposite (v3) direction arrow.
    Yellow: valid plane intersection. Invalid rays get arrows only.
    """
    clear_camera_rays(env)
    shapes = []

    def vector_text(point):
        return ' '.join(f'{float(value):.9g}' for value in point)

    def line_shape(points, indices, color):
        coordinates = ', '.join(vector_text(point) for point in points)
        return f'''Shape {{
            castShadows FALSE
            appearance Appearance {{ material Material {{
                diffuseColor {color} emissiveColor {color}
            }} }}
            geometry IndexedLineSet {{
                coord Coordinate {{ point [ {coordinates} ] }}
                coordIndex [ {indices} ]
            }}
        }}'''

    def arrow(origin, direction, length, color):
        reference = np.array([0., 0., 1.]) if abs(direction[2]) < .9 else np.array([0., 1., 0.])
        side = np.cross(direction, reference)
        side /= np.linalg.norm(side)
        tip = origin + length * direction
        base = tip - .25 * length * direction
        wing = .13 * length * side
        return line_shape([origin, tip, base + wing, base - wing],
                          '0, 1, -1, 2, 1, 3, -1', color)

    for ray in rays:
        origin = np.asarray(ray['origin'], dtype=float)
        direction = np.asarray(ray['direction'], dtype=float)
        intersection = np.asarray(ray['intersection'], dtype=float)
        if (origin.shape != (3,) or direction.shape != (3,)
                or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction))):
            continue
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction = direction / norm
        valid = intersection.shape == (3,) and np.all(np.isfinite(intersection))
        length = min(.18, .5 * np.linalg.norm(intersection - origin)) if valid else .18
        shapes.append(arrow(origin, direction, length, '0 1 1'))
        shapes.append(arrow(origin, -direction, length, '1 0 1'))
        if valid:
            shapes.append(line_shape([origin, intersection], '0, 1, -1', '0 1 1'))
            shapes.append(f'''Transform {{
                translation {vector_text(intersection)}
                children [ Shape {{
                    castShadows FALSE
                    appearance Appearance {{ material Material {{
                        diffuseColor 1 1 0 emissiveColor 1 1 0
                    }} }}
                    geometry Sphere {{ radius 0.006 }}
                }} ]
            }}''')
    if not shapes:
        return
    env.children_field.importMFNodeFromString(
        -1, 'DEF DEBUG_CAMERA_RAYS Transform { children [ ' + '\n'.join(shapes) + ' ] }')
    node = env.supervisor.getFromDef('DEBUG_CAMERA_RAYS')
    if node is not None:
        node.setVisibility(env.camera_node, False)


def clear_discovered_targets(env) -> None:
    """Remove final scan markers independently of the per-query ray overlay."""
    node = env.supervisor.getFromDef('DEBUG_DISCOVERED_TARGETS')
    if node is not None:
        node.remove()


def draw_discovered_targets(env, points) -> None:
    """Keep orange markers at the final estimated positions until reset.

    Points is a mapping of target name to world position. Markers represent
    localization, regardless of classification/verification state, and use no GT.
    """
    clear_discovered_targets(env)
    markers = []
    for name, point in points.items():
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            continue
        position = ' '.join(f'{float(value):.9g}' for value in point)
        marker_id = ''.join(char if char.isalnum() else '_' for char in name)
        markers.append(f'''DEF DEBUG_SCAN_TARGET_{marker_id} Transform {{
            translation {position}
            children [
                Shape {{
                    castShadows FALSE
                    appearance Appearance {{ material Material {{
                        diffuseColor 1 0.25 0 emissiveColor 1 0.25 0
                    }} }}
                    geometry Sphere {{ radius 0.009 }}
                }}
                Shape {{
                    castShadows FALSE
                    appearance Appearance {{ material Material {{
                        diffuseColor 1 0.25 0 emissiveColor 1 0.25 0
                    }} }}
                    geometry IndexedLineSet {{
                        coord Coordinate {{ point [
                            -0.025 0 0, 0.025 0 0,
                            0 -0.025 0, 0 0.025 0,
                            0 0 -0.025, 0 0 0.025
                        ] }}
                        coordIndex [ 0, 1, -1, 2, 3, -1, 4, 5, -1 ]
                    }}
                }}
            ]
        }}''')
    if not markers:
        return
    env.children_field.importMFNodeFromString(
        -1, 'DEF DEBUG_DISCOVERED_TARGETS Transform { children [ '
        + '\n'.join(markers) + ' ] }')
    node = env.supervisor.getFromDef('DEBUG_DISCOVERED_TARGETS')
    if node is not None:
        node.setVisibility(env.camera_node, False)


def draw_workspace_bounds(env) -> None:
        # Remove previous visualization if it exists
        old_node = env.supervisor.getFromDef("DEBUG_WORKSPACE_BOUNDS")
        if old_node is not None:
            old_node.remove()

        min_x, max_x, min_y, max_y, min_z, max_z = env.workspace_bounds

        node_string = f"""
        DEF DEBUG_WORKSPACE_BOUNDS Shape {{
            appearance PBRAppearance {{
                baseColor 1 0 0
                roughness 1
                metalness 0
            }}

            geometry IndexedLineSet {{
                coord Coordinate {{
                point [
                    {min_x} {min_y} {min_z},
                    {max_x} {min_y} {min_z},
                    {max_x} {max_y} {min_z},
                    {min_x} {max_y} {min_z},

                    {min_x} {min_y} {max_z},
                    {max_x} {min_y} {max_z},
                    {max_x} {max_y} {max_z},
                    {min_x} {max_y} {max_z}
                ]
                }}

                coordIndex [
                0, 1, 2, 3, 0, -1,
                4, 5, 6, 7, 4, -1,
                0, 4, -1,
                1, 5, -1,
                2, 6, -1,
                3, 7, -1
                ]
            }}
        }}
        """

        env.children_field.importMFNodeFromString(
            -1,
            node_string
        )

        debug_node = env.supervisor.getFromDef(
            "DEBUG_WORKSPACE_BOUNDS"
        )

        # Visible in Webots main view, invisible to agent camera
        if debug_node is not None:
            debug_node.setVisibility(
                env.camera_node,
                False
            )

def draw_valid_zone(env, screw):
        # Remove previous gate if it exists
        old_gate = env.supervisor.getFromDef("DEBUG_VALID_ZONE")
        if old_gate is not None:
            old_gate.remove()

        radius = 0.1
        height = env.z_max_threshold - env.z_min_threshold
        z_center = 0.5 * (
            env.z_min_threshold + env.z_max_threshold
        )

        x = float(screw.point[0])
        y = float(screw.point[1])

        node_string = f"""
        DEF DEBUG_VALID_ZONE Transform {{
            translation {x} {y} {z_center}
            children [
                Shape {{
                    appearance PBRAppearance {{
                        baseColor 0.1 0.8 0.1
                        transparency 0.5
                        roughness 1
                        metalness 0
                    }}
                    geometry Cylinder {{
                        radius {radius}
                        height {height}
                    }}
                }}
            ]
        }}
        """
        env.children_field.importMFNodeFromString(
            -1,
            node_string
        )
        env.valid_zone_node = env.supervisor.getFromDef("DEBUG_VALID_ZONE")

        # Keep debugging geometry invisible to robot camera
        env.valid_zone_node.setVisibility(env.camera_node, False)        

def draw_agent_zone(env) -> None:
    old_zone = env.supervisor.getFromDef("DEBUG_AGENT_ZONE")
    if old_zone is not None:
        old_zone.remove()

    radius = 0.1
    height = env.z_max_threshold - env.z_min_threshold

    robot_pos = env.get_robot_position()

    x = float(robot_pos[0])
    y = float(robot_pos[1])
    z = float(robot_pos[2])

    node_string = f"""
    DEF DEBUG_AGENT_ZONE Transform {{
        translation {x} {y} {z}
        children [
            Shape {{
                appearance PBRAppearance {{
                    baseColor 0.1 0.3 1.0
                    transparency 0.5
                    roughness 1
                    metalness 0
                }}
                geometry Cylinder {{
                    radius {radius}
                    height {height}
                }}
            }}
        ]
    }}
    """

    env.children_field.importMFNodeFromString(
        -1,
        node_string
    )

    env.agent_zone_node = env.supervisor.getFromDef("DEBUG_AGENT_ZONE")

    if env.agent_zone_node is not None:
        env.agent_zone_translation = (env.agent_zone_node.getField("translation"))

        # Keep debugging geometry invisible to robot camera
        env.agent_zone_node.setVisibility(env.camera_node, False)

def update_agent_zone(env) -> None:
    if getattr(env, "agent_zone_translation", None) is None:
        return

    robot_pos = env.get_robot_position()

    z_center = 0.5 * (
        env.z_min_threshold +
        env.z_max_threshold
    )

    env.agent_zone_translation.setSFVec3f([
        float(robot_pos[0]),
        float(robot_pos[1]),
        float(z_center),
    ])
