"""Reference synthetic trajectory for the cube-in-bowl workshop task."""

import numpy as np

from rospin_workshop.trajectory import EpisodeContext, trajectory


@trajectory(task="cube_in_bowl")
def run(ctx: EpisodeContext) -> None:
    cube = ctx.object_position("cube")
    bowl = ctx.object_position("bowl")

    # A seeded variation prevents every synthetic episode from following the
    # exact same transfer path while keeping the grasp centred on the cube.
    transfer_offset = np.array(
        [ctx.rng.uniform(-0.008, 0.008), ctx.rng.uniform(-0.008, 0.008), 0.0]
    )
    grasp_offset = np.array([0.012, 0.0, -0.004])

    ctx.open_gripper()
    ctx.move_to(
        cube + grasp_offset + [0.0, 0.0, 0.08],
        speed=0.055,
        name="approach_cube",
    )
    # Shoulder pan changes the jaws' world yaw. Counter-rotate the wrist so
    # cubes throughout the randomized workspace are approached with the same
    # grasp orientation instead of being pushed sideways by a jaw.
    grasp_joints = ctx.current_joints
    grasp_joints[4] += grasp_joints[0]
    ctx.move_joints(grasp_joints, speed=0.7, name="align_gripper")
    ctx.move_linear(
        cube + grasp_offset,
        speed=0.025,
        name="descend_to_cube",
    )
    ctx.close_gripper(until_contact=True)
    ctx.move_relative(z=0.10, speed=0.035, name="lift_cube")
    ctx.move_to(
        bowl + transfer_offset + [0.0, 0.0, 0.13],
        speed=0.055,
        safe_height=0.90,
        name="transfer_above_bowl",
    )
    ctx.move_linear(
        bowl + transfer_offset + [0.0, 0.0, 0.125],
        speed=0.025,
        allow_contact=True,
        name="lower_into_bowl",
    )
    ctx.open_gripper()
    ctx.wait(0.4, name="release_cube")
    ctx.move_relative(z=0.10, speed=0.035, name="retreat")
    # Preserve the open gripper so the YAML success predicate can remain true
    # while the arm returns to its home joint configuration.
    ctx.move_home(preserve_gripper=True)
