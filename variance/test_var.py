import drjit as dr
import numpy as np

# utils
Float = dr.llvm.ad.Float


def get_grad(a, loss):
    dr.backward(loss, flags=dr.ADFlag.BackPropGrad | dr.ADFlag.ClearVertices)
    grad = dr.grad(a)
    dr.set_grad(a, 0)
    return grad


def get_sq_grad(a, loss):
    dr.backward(loss, flags=dr.ADFlag.BackPropVar | dr.ADFlag.ClearVertices)
    sq_sum = dr.grad(a)
    dr.set_grad(a, 0)
    return sq_sum


def get_ones(a, loss):
    dr.backward(loss, flags=dr.ADFlag.BackPropOnes | dr.ADFlag.ClearVertices)
    ones = dr.grad(a)
    dr.set_grad(a, 0)
    return ones


def compute_variance(grad, sq_grad, ones):
    return sq_grad - grad * grad / ones


def assert_backprop(a, loss, expected_grad, expected_sq_grad, expected_ones):
    assert np.allclose(get_grad(a, loss), expected_grad)
    assert np.allclose(get_sq_grad(a, loss), expected_sq_grad)
    assert np.allclose(get_ones(a, loss), expected_ones)


def assert_variance(a, loss, expected_var):
    grad = get_grad(a, loss)
    sq_grad = get_sq_grad(a, loss)
    ones = get_ones(a, loss)
    var = compute_variance(grad, sq_grad, ones)
    assert var == expected_var


class Test1D:
    def test_basic(self):
        a = Float(4.0)
        dr.enable_grad(a)

        def f(x):
            return x

        def df(x):
            return 1

        assert_backprop(a, f(a), df(a), df(a) ** 2, 1.0)
        assert_variance(a, f(a), 0.0)

    def test_linear(self):
        a = Float(4.0)
        dr.enable_grad(a)

        def f(x):
            return 2 * x

        def df(x):
            return 2

        assert_backprop(a, f(a), df(a), df(a) ** 2, 1.0)
        assert_variance(a, f(a), 0.0)

    def test_sum(self):
        a = Float(4.0)
        dr.enable_grad(a)

        def f(x):
            return x + x

        def df(x):
            return 1 + 1

        def df_sq_sum(x):
            return 1**2 + 1**2

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 2.0)
        assert_variance(a, f(a), 0.0)

    def test_linear_array(self):
        a = Float(4.0)
        dr.enable_grad(a)

        arr = np.array([1.0, 2.0, 3.0, 4.0])

        def f(x):
            b = x * Float(arr)
            return dr.sum(2.0 * b)

        def df(x):
            return np.sum(2.0 * arr)

        def df_sq_sum(x):
            return np.sum((2.0 * arr) ** 2)

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 4.0)

    def test_linear_array_2(self):
        a = Float(4.0)
        dr.enable_grad(a)
        arr = np.array([1.0, 2.0, 3.0, 4.0])

        def f(x):
            return dr.sum(2.0 * x * Float(arr))

        def df(x):
            return np.sum(2.0 * arr)

        def df_sq_sum(x):
            return np.sum((2.0 * arr) ** 2)

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 4.0)

    def test_linear_array_3(self):
        a = Float(4.0)
        dr.enable_grad(a)
        arr = np.array([1.0, 2.0, 3.0, 4.0])

        def f(x):
            return dr.sum(2.0 * x * Float(arr) * 1.0)

        def df(x):
            return np.sum(2.0 * arr)

        def df_sq_sum(x):
            return np.sum((2.0 * arr) ** 2)

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 4.0)

    def test_product(self):
        a = Float(3.0)
        dr.enable_grad(a)

        def f(x):
            return x * x

        def df(x):
            return x + x

        def df_sq_sum(x):
            return x**2 + x**2

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 2.0)
        assert_variance(a, f(a), 0.0)

    def test_square(self):
        a = Float(3.0)
        dr.enable_grad(a)
        coeff = 2.00001

        def f(x):
            return x**coeff

        def df(x):
            return coeff * x ** (coeff - 1)

        assert_backprop(a, f(a), df(a), df(a) ** 2, 1.0)
        assert_variance(a, f(a), 0.0)


class TestND:
    def test_basic(self):
        a = Float([4.0, 5.0])
        dr.enable_grad(a)
        loss = a
        assert_backprop(a, loss, [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
        assert_variance(a, loss, [0.0, 0.0])

    def test_linear(self):
        a = Float([4.0, 5.0])
        dr.enable_grad(a)
        loss = 2 * a
        assert_backprop(a, loss, [2.0, 2.0], [4.0, 4.0], [1.0, 1.0])
        assert_variance(a, loss, [0.0, 0.0])

    def test_sum(self):
        a = Float([4.0, 5.0])
        dr.enable_grad(a)
        loss = a + a
        assert_backprop(a, loss, [2.0, 2.0], [2.0, 2.0], [2.0, 2.0])
        assert_variance(a, loss, [0.0, 0.0])


if __name__ == "__main__":
    a = Float(4.0)
    dr.enable_grad(a)

    b = a * Float([1, 2, 3, 4])
    # b = 1.0 * a * Float([1, 2, 3, 4])
    # b = a * Float([1])
    c = 2.0 * b
    c = 2.0 * a * Float([1, 2, 3, 4])
    # c = a * Float([1, 2, 3, 4]) + a * Float([1, 2, 3, 4])
    # c = a + a
    c = 2 * a
    # c = a * a
    # c = a**2

    loss = dr.sum(c)

    grad = get_grad(a, loss)
    print(f"grad: {grad}")
    print()

    sq_sum = get_sq_grad(a, loss)
    print(f"sq_sum: {sq_sum}")
    print()

    ones = get_ones(a, loss)
    print(f"ones: {ones}")
    print()

    var = sq_sum - grad * grad / ones
    print(f"var: {var}")
