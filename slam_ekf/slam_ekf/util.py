#!/usr/bin/python3
import numpy as np
from nav_msgs.msg import Odometry
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

def angle_between_yaw(yaw1, yaw2):
    """calculates the angle between two frames
    specified by their yaw angles. Avoid having
    to deal with wrapping the angles by expressing
    frame 2 under frame 1

    Args:
        yaw1 (_type_): yaw angle of the ref frame
        yaw2 (_type_): yaw angle of the query frame/vector

    Returns:
        theta: yaw2 minus yaw1 expressed in yaw1 frame
    """
    s = np.sin(yaw1)
    c = np.cos(yaw1)
    R = np.array([[c,-s],[s,c]])
    p = np.array([np.cos(yaw2),np.sin(yaw2)])[:,np.newaxis]
    p_ = R.T.dot(p) # expressed in the frame of yaw1
    theta = np.arctan2(p_[1,0],p_[0,0])
    return theta

def get_cov_ellipse_pts(mu, cov):
    """return the set of points on a ellipse that
      represents the mean and covariance of the gaussian.
      for plotting purpuses.

    Args:
        mu (_type_): mean of the gaussion
        cov (_type_): covariance of the gaussion

    Returns:
        pts: 2 by n matrix of the points on the ellipse 
        representing the gaussion
    """
    x,y=mu
    # compute eig vector
    W,D = np.linalg.eig(cov)
    # set up the points
    ind = np.flip(np.argsort(W))
    W = np.sqrt(W[ind])
    D = D[:,ind]

    t = np.linspace(0,2*np.pi,30)
    xs = 2*W[0]*np.cos(t) # long axis
    ys = 2*W[1]*np.sin(t) # short axis
    pts = np.hstack((xs[:,np.newaxis],ys[:,np.newaxis]))
    pts = D.dot(pts.T) # rotate
    pts += np.array([[x,y]]).T
    return pts.T

def plot_cov(plot_handle, ax, mu, cov):
    """plot the gaussian as an ellipse

    Args:
        plot_handle (_type_): _description_
        ax : 
        mu (_type_): _description_
        cov (_type_): _description_
    """
    xys = get_cov_ellipse_pts(mu=mu,cov=cov)
    if plot_handle is None:
        plot_handle = \
            ax.plot(xys[:,0],xys[:,1],
                    linestyle='-.',
                    color=(0.2,0.2,0.2),
                    linewidth=0.5)[0]
    else:
        plot_handle.set_data(xys[:,0],xys[:,1])
    
    return plot_handle

def circle_loss(x, data):
    """calculate the loss for fitting the center
    and radius of the cylinder, based on lidar reflection
    points

    Args:
        x (_type_): a 3 by 1 vector of cylinder center (x,y)
        and radius r
        data (_type_): the lidar points expressed as xy coordinates

    Returns:
        out: the loss to be minimized
    """
    # data is 2d array
    out = (data[:,0]-x[0])**2 + (data[:,1]-x[1])**2 - x[2]**2
    out = np.sum(out**2)
    return out

def initial_value(data : np.ndarray):
    """calculates the initial guess of the cylinder center
    and radius

    Args:
        data (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    if data.shape[0] >= 4: # at least 4 points
        n = data.shape[0]
        i1,i2,i3,i4 = np.random.permutation(n)[:4]

        # pick two pairs of points
        x1,y1 = data[i1,:]
        x2,y2 = data[i2,:]
        x3,y3 = data[i3,:]
        x4,y4 = data[i4,:]

        A = np.array([[x1-x2,y1-y2],[x3-x4,y3-y4]])
        b = np.array([0.5*(x1**2-x2**2)+0.5*(y1**2-y2**2),\
                      0.5*(x3**2-x4**2)+0.5*(y3**2-y4**2)])[:,np.newaxis]
        
        c = np.linalg.inv(A).dot(b)
        c : np.ndarray = c.flatten()
        r = np.mean(np.linalg.norm(data-c[np.newaxis,:],axis=1))

        return np.concatenate([c,[r]])
    else:
        return None

def get_center_radius(data : np.ndarray):
    """find the center and radius of the cylinder 
    through minimizing the loss function

    Args:
        data (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    i = 0   
    while i <= 10:
        i+=1
        # compute initial guess
        v0 = initial_value(data)
        if v0 is None:
            return None, None
        out = minimize(circle_loss, x0=v0,args=(data,))
        if out.success is True:
            # check the residue
            c = out.x[:2]
            r = out.x[-1]
            res = np.abs(np.linalg.norm(data-c[np.newaxis,:],axis=1)-r)
            if np.mean(res)<=0.1:
                return c, r

    return None, None

def quaternion_to_yaw(msg : Odometry):
    """extract yaw info from odom messages

    Args:
        msg (Odometry): _description_

    Returns:
        _type_: _description_
    """
    r = Rotation.from_quat([msg.pose.pose.orientation.x,
                            msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w])
    rpy = r.as_euler('xyz')
    return rpy[-1]