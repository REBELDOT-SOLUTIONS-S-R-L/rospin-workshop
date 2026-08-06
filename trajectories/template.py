"""Copy this file, bind it to a task, and replace the example phases."""

from rospin_workshop.trajectory import EpisodeContext, trajectory


@trajectory(task="cube_in_bowl")
def run(ctx: EpisodeContext) -> None:
    # Object positions are evaluated after the seeded task reset.
    cube = ctx.object_position("cube")

    # Every movement starts from the measured robot state. Participants only
    # define the target, path type, and speed.
    ctx.move_to(
        cube + [0.0, 0.0, 0.08],
        speed=0.05,
        name="example_move_above_cube",
    )

    # Available building blocks:
    # ctx.move_linear(position, speed=0.03, allow_contact=True)
    # ctx.move_relative(x=..., y=..., z=...)
    # ctx.move_joints([...])
    # ctx.open_gripper()
    # ctx.close_gripper(until_contact=True)
    # ctx.wait_until_settled("cube")
    # ctx.assert_condition(condition, "explanation")
    ctx.move_home()
