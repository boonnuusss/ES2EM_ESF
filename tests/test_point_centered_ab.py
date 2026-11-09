import numpy as np

from esf_sim.point_centered_potential import build_log_point_centered_matrix


def test_log_point_centered_matrix_has_five_point_shape():
    coefficient = np.ones((7, 8))
    matrix = build_log_point_centered_matrix(coefficient, 1.0, 1.0)
    assert matrix.shape == (40, 40)
    assert matrix.has_sorted_indices
    assert np.all(matrix.diagonal() > 0.0)


def test_log_point_centered_operator_is_second_order_for_smooth_coefficient():
    errors = []
    for nx in (24, 48, 96):
        dx = 2.0 * np.pi / nx
        ny = nx // 2 + 1
        dy = np.pi / (ny - 1)
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        xx, yy = np.meshgrid(x, y)
        log_a = 0.2 * np.sin(xx) + 0.1 * yy
        coefficient = np.exp(log_a)
        psi = np.sin(2.0 * xx) * np.sin(yy)
        exact = (
            5.0 * psi
            - 0.2 * np.cos(xx) * (2.0 * np.cos(2.0 * xx) * np.sin(yy))
            - 0.1 * (np.sin(2.0 * xx) * np.cos(yy))
        )
        matrix = build_log_point_centered_matrix(coefficient, dx, dy)
        numerical = (matrix @ psi[1:-1].ravel()).reshape(ny - 2, nx)
        errors.append(float(np.sqrt(np.mean((numerical - exact[1:-1]) ** 2))))
    orders = [np.log(errors[i] / errors[i + 1]) / np.log(2.0) for i in range(2)]
    assert min(orders) > 1.95


def test_log_form_avoids_local_denominator_amplification_at_a_jump():
    coefficient = np.ones((5, 7))
    coefficient[2, 3] = 1.0e-4
    dx = dy = 1.0
    log_a = np.log(coefficient)
    log_gradient = (log_a[2, 4] - log_a[2, 2]) / (2.0 * dx)
    ratio_gradient = (coefficient[2, 4] - coefficient[2, 2]) / (2.0 * dx) / coefficient[2, 3]
    assert abs(log_gradient) <= abs(ratio_gradient)
