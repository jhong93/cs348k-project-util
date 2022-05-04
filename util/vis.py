from .pose import Pose


LEFT_COLOR = (0.5, 0.8, 1)[::-1]
RIGHT_COLOR = (0.5, 1, 0.8)[::-1]
MID_COLOR = (0.8, 0.5, 1)[::-1]
RACKET_COLOR = (0, 0, 0)
EAR_COLOR = (0, 0.8, 0.8)

LEFT_BONES = [(5, 1), (7, 5), (9, 7), (11, 5), (13, 11), (15, 13)]
RIGHT_BONES = [(6, 1), (8, 6), (10, 8), (12, 6), (14, 12), (16, 14)]
MID_BONES = [(0, 2), (0, 1), (11, 12)]
EAR_BONES = [(3, 0), (4, 0)]

# Additional bones in exported data
LEFT_FEET = [(15, 17), (15, 19)]
RIGHT_FEET = [(16, 18), (16, 20)]
RACKET = [(21, 22)]


def render_2d_pose(ax, keyp, bound=1.):
    assert keyp.shape[0] >= Pose.NumKeypoints
    assert keyp.shape[1] == 2

    for bones, color in (
        (MID_BONES, MID_COLOR),
        (LEFT_BONES, LEFT_COLOR),
        (RIGHT_BONES, RIGHT_COLOR),
        (EAR_BONES, EAR_COLOR),
        (LEFT_FEET, LEFT_COLOR),
        (RIGHT_FEET, RIGHT_COLOR),
        (RACKET, RACKET_COLOR)
    ):
        for i, j in bones:
            if min(i, j) >= keyp.shape[0]:
                continue
            ab = keyp[[i, j], :]
            ax.plot(ab[:, 0], ab[:, 1], lw=2, c=color)

    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('z')