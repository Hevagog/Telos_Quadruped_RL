import pybullet as p


def draw_contact_points(contact_points) -> None:
    for cp in contact_points:
        dot_radius = 0.01
        sphere_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=dot_radius,
            rgbaColor=[
                1.0,
                0.25,
                1.0,
                1.0,
            ],
            specularColor=[0.5, 0.5, 0.5],
        )

        p.createMultiBody(
            baseVisualShapeIndex=sphere_visual,
            basePosition=cp,
        )


def draw_rectangle(points) -> None:
    rectangle_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[
            (points[1][0] - points[0][0]) / 2,
            (points[2][1] - points[0][1]) / 2,
            (points[3][2] - points[0][2]) / 2,
        ],
    )

    rectangle_body = p.createMultiBody(
        baseCollisionShapeIndex=rectangle_shape,
        basePosition=[
            (points[0][0] + points[1][0]) / 2,
            (points[0][1] + points[2][1]) / 2,
            (points[0][2] + points[3][2]) / 2,
        ],
        baseOrientation=[0, 0, 0, 1],
    )


def draw_point(point):
    dot_radius = 0.1
    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=dot_radius,
        rgbaColor=[
            1.0,
            0.25,
            1.0,
            1.0,
        ],
        specularColor=[0.5, 0.5, 0.5],
    )

    p.createMultiBody(baseVisualShapeIndex=sphere_visual, basePosition=point)


def draw_vertical_line(point):
    line_visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=0.01,
        length=2,
        rgbaColor=[1.0, 0.25, 1.0, 1.0],
        specularColor=[0.5, 0.5, 0.5],
    )

    p.createMultiBody(
        baseVisualShapeIndex=line_visual,
        basePosition=point,
    )
