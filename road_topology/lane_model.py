# from scipy.interpolate import lagrange
# from numpy.polynomial.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt

class LaneModel:
    def __init__(self):
        self.a0 = 0
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0

    def load_model(self, a0, a1, a2, a3, a4):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def shift_a0(self, shift):
        self.a0 += shift

    def shift_a1(self, shift):
        self.a1 += shift

    def calc_model_from_points_least_square(self, points, degree):
        # numpy.polynomial.polynomial.polyfit(x, y, deg, rcond=None, full=False, w=None)
        x = points[:, 0]
        y = points[:, 1]

        coeffs = np.polynomial.polynomial.polyfit(x, y, degree)
        assert (degree > 2) and (degree < 5)
        self.a0 = coeffs[0]
        self.a1 = coeffs[1]
        self.a1 = coeffs[1]
        self.a2 = coeffs[2]

        try:
            self.a3 = coeffs[3]
        except:
            pass
        try:
            self.a4 = coeffs[4]
        except:
            pass

    def calc_model_from_points_lagrange(self, points, degree=4):

        x1 = points[:, 0]
        y = points[:, 1]
        x0 = np.ones_like(x1)
        x2 = x1 ** 2
        x_mat = None
        if degree == 4:
            x3 = x2 * x1
            x4 = x2 * x2
            x_mat = np.stack((x0, x1, x2, x3, x4), axis=1)
        elif degree == 2:
            x_mat = np.stack((x0, x1, x2), axis=1)
        else:
            print("degree", degree, "is not supported at the moment")
            return None
        coeffs = np.linalg.solve(x_mat, y)
        self.a0 = coeffs[0]
        self.a1 = coeffs[1]
        self.a2 = coeffs[2]
        if degree == 4:
            self.a3 = coeffs[3]
            self.a4 = coeffs[4]
        elif degree ==2:
            self.a3 = 0
            self.a4 = 0

        # poly = lagrange(x1, y)
        # coeffs = Polynomial(poly).coef
        # # print("coeffs", coeffs)
        # self.a0 = coeffs[-1]
        # self.a1 = coeffs[-2]
        # self.a2 = coeffs[-3]
        # try:
        #     self.a3 = coeffs[-4]
        #     try:
        #         self.a4 = coeffs[-5]
        #     except:
        #         pass
        # except:
        #     pass
    def add_model_3rd_4th_parameters(self, model):
        # self.a0 += model.a0
        # self.a1 += model.a1
        # self.a2 += model.a2
        self.a3 += model.a3
        self.a4 += model.a4

    def calc_cubic_model_from_points_fixed_dydx_at_x0(self, points, dydx_at_x_dot, x_dot):
        x1 = points[:, 0]
        y = points[:, 1]
        x0 = np.ones_like(x1)
        x2 = x1 ** 2
        #x5 = x2 * x3
        x_mat = np.stack((x0, x1, x2), axis=1)
        # untill here we dealt witht the 4 equation from usual type data, now we want to add the 5th equation,
        # which is looking like dxdy = 0 + a1 + 2 * a2 * x0 + 3 * a3 * x0**2 + 4a4 * x0**3
        x_extra_line = np.expand_dims(np.asarray([0, 1, 2 * x_dot]), axis=0)
        y_extra_line = np.asarray([dydx_at_x_dot])
        x_mat = np.concatenate((x_mat, x_extra_line), axis=0)
        y = np.concatenate((y, y_extra_line))
        coeffs = np.linalg.solve(x_mat, y)
        self.a0 = coeffs[0]
        self.a1 = coeffs[1]
        self.a2 = coeffs[2]
        self.a3 = 0
        self.a4 = 0

    def calc_quadratic_model_from_points_lagrange_fixed_dydx_at_x0(self, points, dydx_at_x_dot, x_dot):
        x1 = points[:, 0]
        y = points[:, 1]
        x0 = np.ones_like(x1)
        x2 = x1 ** 2
        x3 = x2 * x1
        x4 = x2 * x2
        x_mat = np.stack((x0, x1, x2, x3, x4), axis=1)

        # untill here we dealt witht the 4 equation from usual type data, now we want to add the 5th equation,
        # which is looking like dxdy = 0 + a1 + 2 * a2 * x0 + 3 * a3 * x0**2 + 4a4 * x0**3
        y_extra_line = np.asarray([dydx_at_x_dot])

        x_extra_line = np.expand_dims(np.asarray([0, 1, 2 * x_dot, 3 * x_dot ** 2, 4 * x_dot ** 3]), axis=0)
        x_mat = np.concatenate((x_mat, x_extra_line), axis=0)
        y = np.concatenate((y, y_extra_line))
        coeffs = np.linalg.solve(x_mat, y)
        self.a0 = coeffs[0]
        self.a1 = coeffs[1]
        self.a2 = coeffs[2]
        self.a3 = coeffs[3]
        self.a4 = coeffs[4]

    def flip_back_front(self, dist=100):
        self.a0 *= -1
        self.a1 *= -1
        self.a2 *= -1
        self.a3 *= -1
        self.a4 *= -1

        # self.a0 -= self.Z2X(100)
        # self.a1 -= self.dX_dZ(100)

    def all(self):
        print('a0:', self.a0, 'a1:', self.a1, 'a2:', self.a2, 'a3:', self.a3, 'a4:', self.a4)

    def Z2X(self, Z):
        X = self.a0 + (self.a1 + (self.a2 + (self.a3 + self.a4 * Z) * Z) * Z) * Z
        return X

    def dX_dZ(self, Z):
        return self.a1 + (2 * self.a2 + (3 * self.a3 + 4 * self.a4 * Z) * Z) * Z

    def set_a0(self, new_a0):
        self.a0 = new_a0

    def get_points(self, view_range=None):
        points = list()
        if view_range is None:
            view_range = [-100, 200]

        for Z in range(view_range[0], view_range[1]):
            X = self.Z2X(Z)
            points.append([X, Z])
        np_points = np.asarray(points)
        return np_points

    def display(self, view_range=None):
        points = list()
        if view_range is None:
            view_range = [-100, 200]
        for Z in range(view_range[0], view_range[1]):
            X = self.Z2X(Z)
            points.append([X,Z])
        np_points = np.asarray(points)

        plt.plot(np_points[:, 0])

        plt.show()

if __name__ == "__main__":

    lm = LaneModel()
    lm.load_model(a0=2, a1=0.1, a2=0.001, a3=-0.0006, a4=0)
    lm.display()
    lm.flip_back_front()
    print("b points")
