import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def skew_symmetric(v):
    return np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
    )

class Quaternion():
    def __init__(self, w=1., x=0., y=0., z=0., axis_angle=None, euler=None):
        if axis_angle is None and euler is None:
            self.w = w; self.x = x; self.y = y; self.z = z

        elif axis_angle is not None:
            axis_angle = np.array(axis_angle)
            norm = np.linalg.norm(axis_angle)
            self.w = np.cos(norm / 2)
            if norm < 1e-10:
                self.x = 0; self.y = 0; self.z = 0
            else:
                imag = axis_angle / norm * np.sin(norm / 2)
                self.x = imag[0].item(); self.y = imag[1].item(); self.z = imag[2].item()
        else:
            roll  = euler[0]; pitch = euler[1]; yaw   = euler[2]

            cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
            cr = np.cos(roll * 0.5); sr = np.sin(roll * 0.5)
            cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)

            self.w = cr * cp * cy + sr * sp * sy
            self.x = sr * cp * cy - cr * sp * sy
            self.y = cr * sp * cy + sr * cp * sy
            self.z = cr * cp * sy - sr * sp * cy

    def to_mat(self):
        v = np.array([self.x, self.y, self.z]).reshape(3,1)
        return (self.w ** 2 - v.T @ v) * np.eye(3) + \
               2 * v @ v.T + 2 * self.w * skew_symmetric(v)

    def quat_mult(self, q):
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:]   = -v[:,0]
        sum_term[1:, 0]  = v[:,0]
        sum_term[1:, 1:] = -skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        return sigma @ q

sys.path.append('./data')
with open('data/data.pickle', 'rb') as file:
    data = pickle.load(file)

gt_p     = data[0]
gt_v_ini = data[1]
gt_r_ini = data[2]
imu_a    = data[3]
imu_a_t  = data[4]
imu_w    = data[5]
imu_w_t  = data[6]
gnss     = data[7]
gnss_t   = data[8]

var_imu_a = 0.01
var_imu_w = 0.01
var_gnss = 0.1
gravity = 9.81

g = np.array([0, 0, -gravity])
lJac = np.zeros([9, 6])         # motion model noise jacobian
lJac[3:, :] = np.eye(6)
hJac = np.zeros([3, 9])         # measurement model jacobian
hJac[:, :3] = np.eye(3)

pEst = np.zeros([imu_a.shape[0], 3])     # position estimates
vEst = np.zeros([imu_a.shape[0], 3])     # velocity estimates
qEst = np.zeros([imu_a.shape[0], 4])     # orientation estimates as quaternions
pCov = np.zeros([imu_a.shape[0], 9, 9])  # covariance matrices

# Initial values
pEst[0] = gt_p[0]
vEst[0] = gt_v_ini
q_gt_r_ini = Quaternion(euler=gt_r_ini)
qEst[0] = np.array([q_gt_r_ini.w, q_gt_r_ini.x, q_gt_r_ini.y, q_gt_r_ini.z])
pCov[0] = np.eye(9)

def predict_update(pCovTemp, pTemp, vTemp, qTemp, delta_t, imu_a,  imu_w):

    Rotation_Mat = Quaternion(*qTemp).to_mat()
    pEst = pTemp + delta_t * vTemp + 0.5 * (delta_t ** 2) * (Rotation_Mat @ imu_a - g)
    vEst = vTemp + delta_t * (Rotation_Mat @ imu_a - g)
    qEst = Quaternion(euler = delta_t * imu_w).quat_mult(qTemp)

    F = np.eye(9)
    imu = imu_a.reshape((3, 1))
    F[0:3, 3:6] = delta_t * np.eye(3)
    F[3:6, 6:9] = Rotation_Mat @ (-skew_symmetric(imu)) * delta_t

    Q = np.eye(6)
    Q[0:3, 0:3] = var_imu_a * Q[0:3, 0:3]
    Q[3:6, 3:6] = var_imu_w * Q[3:6, 3:6]
    Q = Q * (delta_t ** 2)
    pCov = F @ pCovTemp @ F.T + lJac @ Q @ lJac.T
    return pEst, vEst, qEst, pCov

def measurement_update(pCovTemp, y_k, pTemp, vTemp, qTemp):

    RCov = var_gnss * np.eye(3)
    K = pCovTemp @ hJac.T @ np.linalg.inv(hJac @ pCovTemp @ hJac.T + RCov)

    delta_x = K @ (y_k - pTemp)

    pTemp = pTemp + delta_x[:3]
    vTemp = vTemp + delta_x[3:6]
    qTemp = Quaternion(axis_angle = delta_x[6:]).quat_mult(qTemp)

    pCovTemp = (np.eye(9) - K @ hJac) @ pCovTemp

    return pTemp, vTemp, qTemp, pCovTemp

def main():
    print(__file__ + " start!!")

    i_gnss = 0

    hz = np.zeros((3,1))
    est_traj_fig = plt.figure()
    ax = est_traj_fig.add_subplot(111, projection='3d')


    for k in range(1, imu_a.shape[0]):

        delta_t = imu_a_t[k] - imu_a_t[k - 1]
        pEst[k], vEst[k], qEst[k], pCov[k] \
            = predict_update(pCov[k-1], pEst[k-1], vEst[k-1], qEst[k-1],
                                                                         delta_t,imu_a[k - 1],imu_w[k - 1])

        if i_gnss < gnss_t.shape[0] and abs(gnss_t[i_gnss] - imu_a_t[k]) < 0.01:
            pEst[k], vEst[k], qEst[k], pCov[k] = measurement_update(pCov[k],
                                                        gnss[i_gnss], pEst[k], vEst[k], qEst[k])
            hz = np.hstack((hz, gnss[i_gnss].reshape(3,1)))
            i_gnss += 1

        if (k % 50 == 0):
            plt.cla()
            ax.plot(hz[0,:], hz[1,:], hz[2,:], ".g", label='GPS')
            ax.plot(pEst[:,0], pEst[:,1], pEst[:,2], '.r', label='Estimated', markersize=1)
            ax.plot(gt_p[:,0], gt_p[:,1], gt_p[:,2], '-b', label='Ground Truth')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')
            ax.set_title('Estimated Trajectory')
            ax.legend()
            ax.set_zlim(-1, 1)
            plt.pause(0.01)

if __name__ == '__main__':
            main()
