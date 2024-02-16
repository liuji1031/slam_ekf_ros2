#!/usr/bin/python3

import numpy as np
from easydict import EasyDict as edict
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import DBSCAN
from .util import quaternion_to_yaw, get_center_radius,\
                  angle_between_yaw, plot_cov

class SlamEkf(Node):
    def __init__(self):
        super().__init__('ekf_slam')

        # for grouping lidar scan points
        self.dbscan = DBSCAN(eps=0.2,min_samples=1)
        
        # subscriber to odom topic, use odom as control input
        self.odom_sub = self.create_subscription(Odometry, '/odom',
                                                 self.odom_callback,10)
        
        # append new odom info to a list before SLAM update
        self.robot_pose_odom = []

        # subscriber to laser scan
        self.scan_sub = self.create_subscription(LaserScan, "/scan",
            self.scan_callback,qos_profile=qos_profile_sensor_data)
        
        self.landmark_measurements = edict(data=None,timestamp=None)
        self.lidar_pts_fixedframe = None
        
        cb_group = MutuallyExclusiveCallbackGroup()
        # create a timer for SLAM updates
        self.create_timer(0.01, self.slam,
                          callback_group=cb_group)
        self.slam_last_ts = None
        
        # initialize robot pose state belief
        # the landmarks are added as the robot observes them
        self.mu = np.zeros((3,1))
        self.sigma = np.zeros((3,3))

        # coefficient for motion noise covariance
        self.alpha = [0.005,0.005,0.1,0.1] # rot/tr/rot/tr
        # coefficient for the sensor noise variance
        self.beta = [0.01,0.001]

        self.landmark_count = 0

        # create plot to show the slam process
        self.fig = plt.figure(figsize=(17,10),constrained_layout=True)
        ncol = 3
        gs = GridSpec(2, ncol, figure=self.fig)
        self.ax1 = self.fig.add_subplot(gs[:, :(ncol-1)],frameon=False)
        self.ax1.set_aspect('equal')
        self.ax2 = self.fig.add_subplot(gs[0, ncol-1],frameon=False)
        self.ax2.set_aspect('equal')
        self.fig.show()
        
        # store handles for all the plots
        self.robot_plot = None
        self.robot_cov_plot = None
        self.landmark_plot = edict()
        self.landmark_measurement_plot = None
        self.lidar_plot = None
        self.cov_mat_plot = None
        self.odom_plot = None
        self.robot_traj_slam = None
        self.robot_traj_odom = None
        # timer for updating plot
        self.create_timer(0.1, self.plot_callback)

    @property
    def robot_state(self):
        """returns the mean of the robot pose

        Returns:
            _type_: _description_
        """
        return self.mu[:3]
    
    def update_robot_state(self, new_state):
        """update the mean of robot pose

        Args:
            new_state (_type_): _description_
        """
        self.mu[:3] = new_state
    
    @property
    def landmark_state(self):
        """extract the landmark xy location from state variable
        """
        return self.mu[3:].reshape((-1,2))
    
    @property
    def sigma_r(self):
        """robot pose covariance

        Returns:
            _type_: _description_
        """
        return self.sigma[:3,:3]

    @property
    def sigma_rm(self):
        """cov between robot pose and landmark

        Returns:
            _type_: _description_
        """
        return self.sigma[:3,3:]
    
    @property
    def sigma_mr(self):
        """cov between landmark and robot pose

        Returns:
            _type_: _description_
        """
        return self.sigma[3:,:3]
    
    @property
    def sigma_m(self):
        """cov between landmarks

        Returns:
            _type_: _description_
        """
        return self.sigma[3:,3:]
    
    def odom_callback(self, msg : Odometry):
        """callback for the odom topic, stores the latest odom info
        in between SLAM updates

        Args:
            msg (Odometry): _description_
        """
        # extract pose info
        yaw = quaternion_to_yaw(msg)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_pose_odom.append([x,y,yaw])

    def scan_callback(self, msg : LaserScan):
        """callback for lidar scan topic; group lidar points using dbscan
        and infer the center and radius of the cylinder, which serves as 
        the landmarks

        Args:
            msg (LaserScan): _description_
        """
        # get angles of the scan
        nscan = round((msg.angle_max - msg.angle_min)/msg.angle_increment)+1
        angles = np.linspace(msg.angle_min, msg.angle_max, nscan, endpoint=True)

        # convert to fixed frame
        x,y,yaw = self.robot_state[:,0]
        rng = np.array(msg.ranges)
        keep = (rng>=msg.range_min) & (rng<=msg.range_max)
        pt_x = x + rng[keep] * np.cos(yaw + angles[keep])
        pt_y = y + rng[keep] * np.sin(yaw + angles[keep])
        pts = np.hstack([pt_x[:,np.newaxis],pt_y[:,np.newaxis]])
        self.lidar_pts_fixedframe = pts # store for plotting

        # cluster points
        pts_clu = self.cluster_pts(pts)
        
        # find the center and radius for each cluster
        all_center = []
        for clu in pts_clu:
            center, r = get_center_radius(clu)
            if center is not None:
                all_center.append(center)
        if len(all_center)==0:
            return
        all_center = np.array(all_center)

        # convert back to range and bearing
        data = []
        for center in all_center:
            dx = center[0]-x
            dy = center[1]-y
            # range and bearing, expressed in robot frame
            angle = angle_between_yaw(yaw, np.arctan2(dy,dx))
            data.append([np.sqrt(dx**2+dy**2),angle])
        data = np.array(data)
        self.landmark_measurements.data = data
        self.landmark_measurements.timestamp = self.get_curr_time()

    def plot_callback(self):
        """the timer callback for updating the plot
        """
        # plot robot pose
        x,y,yaw = self.robot_state[:,0]
        d = 0.3
        if self.robot_plot is None:
            # set up new plot
            self.robot_plot = [self.ax1.plot(x,y,label='robot',
                                        ms=6,color='b',marker='o',ls='')[0],
                               self.ax1.plot([x,x+d*np.cos(yaw)],
                                        [y,y+d*np.sin(yaw)],
                                        color='r')[0], # x axis, body frame
                               self.ax1.plot([x,x+d*np.cos(yaw+np.pi/2)],
                                        [y,y+d*np.sin(yaw+np.pi/2)],
                                        color='g')[0] # y axis
                            ]
        else:
            # update data only
            self.robot_plot[0].set_data(x,y)
            self.robot_plot[1].set_data([x,x+d*np.cos(yaw)],
                                        [y,y+d*np.sin(yaw)])
            self.robot_plot[2].set_data([x,x+d*np.cos(yaw+np.pi/2)],
                                        [y,y+d*np.sin(yaw+np.pi/2)])

        # robot pose cov
        self.robot_cov_plot = plot_cov(plot_handle=self.robot_cov_plot,
                                       ax=self.ax1,
                                       mu=self.robot_state[:2,0],
                                       cov=self.sigma[:2,:2])

        # landmark lidar scans
        if self.lidar_plot is None:
            if self.lidar_pts_fixedframe is not None and \
                len(self.lidar_pts_fixedframe)>0:
                self.lidar_plot = self.ax1.plot(self.lidar_pts_fixedframe[:,0],
                                        self.lidar_pts_fixedframe[:,1],
                                        label='lidar',
                                        ms=1,
                                        color=(0.8,0.8,0.8),
                                        marker='o',
                                        ls='')[0]
        else:
            self.lidar_plot.set_data(self.lidar_pts_fixedframe[:,0],
                                     self.lidar_pts_fixedframe[:,1])
            
        # landmarks
        for k,l in enumerate(self.landmark_state):
            indices = [3+2*k,3+2*k+1]
            mu = self.landmark_state[k,:]
            sigma = self.sigma[np.ix_(indices,indices)]
            name = f'landmark_{k:d}'
            if name not in self.landmark_plot:
                p1 = self.ax1.plot(l[0],l[1],
                            ms=6,color='r',marker='+',ls='')[0]
                p2 = plot_cov(plot_handle=None,
                              ax=self.ax1,
                              mu=mu,
                              cov=sigma)
                self.landmark_plot[name] = [p1,p2]
            else:
                # update
                self.landmark_plot[name][0].set_data(l[0],l[1])
                self.landmark_plot[name][1] = plot_cov(
                    plot_handle=self.landmark_plot[name][1],
                    ax=self.ax1,
                    mu=mu,
                    cov=sigma)
        
        # plot odom coordinates
        if len(self.robot_pose_odom)>0:
            odom_x, odom_y = self.robot_pose_odom[-1][:2]
        else:
            odom_x = 0.0
            odom_y = 0.0

        if self.odom_plot is None:
            self.odom_plot = self.ax1.plot(odom_x,odom_y,label='robot_odom',
                        ms=6,color='b',marker='s',ls='',mfc='none')[0]
        else:
            self.odom_plot.set_data(odom_x,odom_y)

        if self.cov_mat_plot is None:
            plt.sca(self.ax2)
            self.cov_mat_plot = plt.imshow(self.sigma, cmap='cool')
            plt.colorbar(ax=self.ax2,aspect=20,shrink=0.3)
        else:
            m = self.sigma.shape[0]
            self.cov_mat_plot.set(data=self.sigma,extent=(0,m+1,0,m+1))
            # plt.colorbar(ax=self.ax2)


        self.ax1.set(xlim=(-3,15),ylim=(-15,3))
        # self.ax2.set(xlim=(0,self.mu.shape[0]),ylim=(0,self.mu.shape[0]))
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()

    def cluster_pts(self, pts):
        """cluster lidar points using dbscan

        Args:
            pts (_type_): _description_

        Returns:
            result: list of point clusters
        """
        if len(pts)==0:
            return []
        clu = self.dbscan.fit(pts)
        labels = clu.labels_
        nclu = np.max(labels)
        result = []
        for i in range(nclu+1):
            result.append(pts[labels==i])
        return result

    def get_curr_time(self):
        """returns the current ros time

        Returns:
            _type_: _description_
        """
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        return float(sec) + float(nanosec)/1.0e9
    
    def slam_timing(self):
        """controls the timing of the slam updates. the timing of updates are 
        controlled by /scan topic publication rate when scan data are available
        otherwise, the rate defaults to 5 Hz, but only performs the prediction 
        step
        Returns:
            proceed: whether to proceed with slam update
            pred_only: whether to perform only the prediction step
        """
        timeout_dur = 0.21 # duration for checking scan data availability
        curr_ts = self.get_curr_time()
        if self.slam_last_ts is None:
            # initialize timestamp
            self.slam_last_ts = self.get_curr_time()

        pred_only = True # default value
        proceed = True
        use_own_timing = True
        landmark_measurements = self.landmark_measurements
        if landmark_measurements.timestamp is not None:
            # have received scan data before
            if landmark_measurements.timestamp > self.slam_last_ts+1e-3:
                # new sensor data
                pred_only = False
                use_own_timing = False
                self.slam_last_ts = landmark_measurements.timestamp
            else:
                if curr_ts >= self.slam_last_ts + timeout_dur:
                    # passed timeout period, no new data
                    use_own_timing = True
                else:
                    # within wait period
                    proceed = False
                    return proceed, pred_only
        else: # no sensor data ever
            use_own_timing = True
        
        if use_own_timing:
            # no new sensor data
            if curr_ts - self.slam_last_ts >= 0.2:
                # update around 5 Hz
                pred_only = True
                self.slam_last_ts = curr_ts
            else:
                proceed = False
                return proceed, pred_only
        return proceed, pred_only
    
    def odom_to_control(self):
        """extract control information from odom, i.e., rotation 1, translation,
        rotation 2

        Returns:
            u: vector containing the control command extraced from odom
        """
        # unpack odom info
        x0,y0,yaw0 = self.robot_pose_odom[0] # initial odom
        x1,y1,yaw1 = self.robot_pose_odom[-1] # latest odom

        # self.get_logger().info(
        #  f"Odom: {x0:.2f},{y0:.2f},{yaw0:.2f}; {x1:.2f},{y1:.2f},{yaw1:.2f}")

        # pop all poses except for the latest
        for _ in range(len(self.robot_pose_odom)-1):
            self.robot_pose_odom.pop(0)

        # convert to odom control input:
        # 1. rotate
        # 2. move forward
        # 3. rotate again
        
        tr = np.sqrt((x1-x0)**2+(y1-y0)**2) # translation
        if tr<0.05:
            # ignore the first rotation if movement is small, as the calculation
            # will be noisy
            yaw0_ = yaw0
        else:
            yaw0_ = np.arctan2(y1-y0, x1-x0)

        # calculate the delta rotation needed to go from one yaw to the second
        rot1 = angle_between_yaw(yaw1=yaw0, yaw2=yaw0_)
        rot2 = angle_between_yaw(yaw1=yaw0_,yaw2=yaw1)

        u = [rot1, tr, rot2]
        return u

    def motion_jacobian(self, x, u):
        """the jacobian of state transition w.r.t 
        previous state

        Args:
            x (_type_): _description_
            u (_type_): _description_

        Returns:
            _type_: _description_
        """
        yaw = x[-1,0]
        rot1,tr,rot2 = u
        phi = yaw+rot1
        J = np.array([[1.0, 0.0, -tr*np.sin(phi)],
                      [0.0, 1.0,  tr*np.cos(phi)],
                      [0.0, 0.0,  1.0]])
        return J
    
    def noise_jacobian(self, x, u):
        """computes how the noise in the command signal transform into
        the state space

        Args:
            x (_type_): robot pose
            u (_type_): control
        
        Returns:
            Jn: noise jacobian
            Rn: the covariance matrix of the control input
        """
        rot1,tr,rot2 = u
        yaw = x[-1,0]

        # construct the noise variance matrix
        # the form of the variance is similar to Probabilistic Robotics
        # book chapter 5.4; set to be proportional to the command signal
        # magnitude
        Rn = np.zeros((3,3))
        a1,a2,a3,a4 = self.alpha
        Rn[0,0] = a1*rot1**2 + a2*tr**2
        Rn[1,1] = a3*(rot1**2 + rot2**2) + a4*tr**2
        Rn[2,2] = a1*rot2**2 + a2*tr**2

        phi = yaw + rot1
        s = np.sin(phi)
        c = np.cos(phi)
        Jn = np.array([[-tr*s,   c, 0.0],
                       [ tr*c,   s, 0.0],
                       [  1.0, 0.0, 1.0]])
        
        return Jn, Rn
    
    def motion_model(self, x, u):
        """motion model based on odometry

        Args:
            x (_type_): _description_
            u (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, y, yaw = x[:,0]
        rot1,tr,rot2 = u
        phi = yaw+rot1
        return np.array([[x + tr*np.cos(phi)],
                         [y + tr*np.sin(phi)],
                         [yaw + rot1 + rot2]])

    def compute_cov_pred(self,
                         J_motion : np.ndarray,
                         J_noise : np.ndarray,
                         Rn : np.ndarray):
        """computes the covariance matrix of the predicted belief
        Args:
            J_motion (np.ndarray): _description_
            J_noise (np.ndarray): _description_
            Rn (np.ndarray): _description_
        """
        # only update portions of the covariance matrix
        # first compute the updated robot pose covariance matrix
        # update in pace
        tmp1 = np.linalg.multi_dot((J_motion, self.sigma_r, J_motion.T))
        tmp2 = np.linalg.multi_dot((J_noise, Rn, J_noise.T))
        self.sigma[:3,:3] = tmp1+tmp2

        if self.sigma.shape[0]>3: # landmarks have been observed
            # now update the covariance between robot pose and landmark location
            tmp = np.linalg.multi_dot((self.sigma_mr,J_motion.T))
            self.sigma[3:,:3] = tmp
            self.sigma[:3,3:] = tmp.T

    def convert_to_fixed_frame(self, landmark_measurements):
        """conver the landmark measurements (range and bearing)
        into xy location in the fixed frame

        Args:
            landmark_measurements (_type_): _description_

        Returns:
            _type_: _description_
        """
        x,y,yaw = self.robot_state[:,0]
        landmark_xy = []
        for p in landmark_measurements:
            x_ = x + p[0]*np.cos(yaw + p[1])
            y_ = y + p[0]*np.sin(yaw + p[1])
            landmark_xy.append([x_,y_])
        return np.array(landmark_xy)

    def find_association(self, l):
        """find the association between the detected landmark
        and the existing landmarks using heuristic based on distance
        Args:
            l (_type_): xy coordinate of the "new" landmark

        Returns:
            j: the index of the existing landmark. if no landmark matches,
            returns -1
        """
        # if some landmark is within the threshold, then consider it
        # the same landmark
        for j, ls in enumerate(self.landmark_state):
            if np.linalg.norm(ls-l)<1.0: 
                return j
        return -1
    
    def sensor_cov(self, r):
        """returns the cov of the range and bearing sensor

        Args:
            r (_type_): the range measurement
        """
        b1, b2 = self.beta
        return np.array([[b1*r,.0],[.0, b2]])

    def initialize_landmark(self, l_xy, measurement):
        """initialize the new landmark's mean and covariance

        Args:
            l_xy (_type_): xy coordinate in fixed world frame
            measurement (_type_): range and bearing
        """
        rng, bearing = measurement
        # l is the xy coordinate in fixed world frame
        self.mu = np.vstack((self.mu, l_xy.reshape((2,1))))

        # now compute the covariance
        x,y,yaw = self.robot_state[:,0]

        # compute jacobian of landmark xy w.r.t robot state
        t = yaw+bearing
        c = np.cos(t)
        s = np.sin(t)
        J1 = np.array([[1.,0.,-rng*s],
                       [0.,1., rng*c]])
        
        # compute jacobian of landmark xy w.r.t r and b
        J2 = np.array([[c, -rng*s],
                       [s,  rng*c]])
        
        # measurement covariance of range and bearing sensor
        Rs = self.sensor_cov(rng)
        
        cov_ll = np.linalg.multi_dot((J1,self.sigma_r,J1.T)) + \
                 np.linalg.multi_dot((J2,Rs,J2.T)) # 2 by 2 matrix
        cov_lx = np.linalg.multi_dot((J1, self.sigma[:3,:])) # 2 by xx matrix

        # expand original covariance matrix
        self.sigma = np.vstack((self.sigma, cov_lx))
        self.sigma = np.hstack((self.sigma, np.vstack((cov_lx.T,cov_ll))))

        self.landmark_count+=1

        self.get_logger().info(
             (f"✅ New landmark added at ({l_xy[0]:.02f},{l_xy[1]:.02f})."
             f"Current total landmark number: {self.landmark_count}"
            ))

    def compute_obs(self, landmark_ind, rng):
        """compute the convariance matrix of the predicted
        measurment and other related variables

        Args:
            landmark_ind (_type_): _description_
            rng (_type_): range measurement

        Returns:
            H: measurement model jacobian
            Z: measurement covariance matrix
            z_pred: predicted measurements

        """
        # get the cov of robot pose and the landmark
        j = landmark_ind
        indices = [0,1,2,3+2*j,3+2*j+1]
        # slice the portion corresponding to robot state
        # and jth landmark within the full covariance matrix
        cov_ = self.sigma[np.ix_(indices,indices)]

        xr,yr,yaw = self.robot_state[:,0]
        xl,yl = self.landmark_state[j,:]
        dx, dy = (xl-xr), (yl-yr) # difference in x and y

        # define some helper variables to save computation
        rho = dx**2+dy**2
        rho_inv = 1.0/rho
        rho_inv_sqrt = np.sqrt(rho_inv)
        c = np.cos(yaw)
        s = np.sin(yaw)
        dx_r = c*dx+s*dy
        dy_r = -s*dx+c*dy
        rho_x = -dy_r/(dx_r**2+dy_r**2)
        rho_y =  dx_r/(dx_r**2+dy_r**2)

        # jacobian w.r.t robot state and jth landmark
        # row 1, range
        h00 = -rho_inv_sqrt*dx
        h01 = -rho_inv_sqrt*dy
       
        # row 2, bearing
        h10 = rho_x*(-c) + rho_y*s
        h11 = rho_x*(-s) + rho_y*(-c)
        h12 = rho_x*(dy_r) + rho_y*(-dx_r)
        H = np.array([[h00, h01, 0.0, -h00, -h01],
                      [h10, h11, h12, -h10, -h11]])

        # computes the predicted measurement
        # to compute predicted angle, use change of coordinate system
        s = np.sin(yaw)
        c = np.cos(yaw)
        R = np.array([[c,-s],[s,c]])
        xl_r,yl_r = R.T.dot(np.array([dx,dy])[:,np.newaxis]).flatten()
        # calculate angle using landmark xy coordinate in robot frame
        angle_pred = np.arctan2(yl_r,xl_r)

        # add sensor covariance to Z
        Z = np.linalg.multi_dot((H, cov_, H.T)) + self.sensor_cov(rng)
        z_pred = np.array([np.sqrt(rho), angle_pred])

        return H, Z, z_pred

    def slam(self):
        """perform actual slam updates
        """
        # check the timing of slam updates
        proceed, pred_only = self.slam_timing()
        if not proceed:
            return
        # self.get_logger().info((f'Slam ts: {self.slam_last_ts:.2f}'
        #                         f'pred only = {pred_only}')
        #                        )
                
        # get control input from the list of robot poses
        if len(self.robot_pose_odom)==0:
            # return if not receiving msg from /odom frame
            return
        
        # extract control signal
        u = self.odom_to_control()

        # motion jacobian, how the next state relates the current state
        J_motion = self.motion_jacobian(x=self.robot_state,u=u)
        
        # compute noise jacobian, i.e., how the noise in the control
        # command translate into the next state
        J_noise, Rn = self.noise_jacobian(x=self.robot_state,u=u)
        
        # ==================  1. prediction step: ==============================
        # the mean of the landmarks do not change
        self.update_robot_state(
            self.motion_model(x=self.robot_state,u=u)
            )
        
        # compute covariance of predicted belief
        self.compute_cov_pred(J_motion, J_noise, Rn)

        if pred_only is True:
            return

        # ==================  2. correction step: ==============================
        # loop through all newly discovered landmarks

        # convert the measurement (polar coordinates) to xy coordinates
        l_measurement_polar = self.landmark_measurements.data
        l_measurement_xy = self.convert_to_fixed_frame(l_measurement_polar)

        # get the landmark in state variable
        for z, l_xy in zip(l_measurement_polar, l_measurement_xy):

            # first find association
            j = self.find_association(l_xy)

            if j==-1: # no association found
                # initialize landmark
                self.initialize_landmark(l_xy,measurement=z)
                
                # set j to the new landmark ind
                j = self.landmark_count-1

            # compute observation model jacobian (H), measurement covariance (Z)
            # , and predicted measurement (z_pred)
            H,Z,z_pred = self.compute_obs(landmark_ind=j,rng=z[0])

            # computes kalman gain
            ind = [0,1,2,3+2*j,3+2*j+1] # indices of robot pose and landmark j
            cov_ = self.sigma[:,ind] # get portions of the cov matrix
            K = np.linalg.multi_dot((cov_,H.T,np.linalg.inv(Z))) # (3+2m) by 2
            
            # compute innovation
            innovation = (z-z_pred)
            innovation[1] = angle_between_yaw(yaw1=z_pred[1],yaw2=z[1])
            innovation = innovation[:,np.newaxis]

            # perform correction step
            self.mu = self.mu + K.dot(innovation)
            self.sigma = self.sigma - np.linalg.multi_dot((K,Z,K.T))

def main(args=None):
    rclpy.init(args=args)
    ekf_slam = SlamEkf()
    ekf_slam.get_logger().info("EKF SLAM started!")
    rclpy.spin(ekf_slam)
    rclpy.shutdown()

if __name__ == "__main__":
    main()