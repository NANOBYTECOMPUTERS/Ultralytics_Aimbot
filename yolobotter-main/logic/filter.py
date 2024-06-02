# Initialize Kalman Filter
kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state variables (x, y, vx, vy), 2 measurements (x, y)
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])  # State transition matrix (constant velocity model)
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])  # Measurement function
kf.P *= 1000  # Initial covariance (high uncertainty)
kf.R = 5  # Measurement noise
kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.1)  # Process noise

# One Euro Filter parameters
min_cutoff = 0.001  # Minimum cutoff frequency for the filter
beta = 0.7  # Filter aggressiveness (0-1)
d_cutoff = 1.0  # Cutoff frequency for the derivative

# Sample noisy point measurements
measurements = [(100, 50), (105, 53), (112, 58), (118, 62)]

# Track the point
for measurement in measurements:
    z = np.array(measurement).reshape(-1, 1)  # Measurement
    kf.predict()
    kf.update(z)

    # Apply One Euro Filter for smoothing
    x_hat, y_hat = kf.x[:2]  # Predicted position
    dx, dy = kf.x[2:]  # Predicted velocity
    
    # Apply one euro filter to x_hat, y_hat using dx, dy, min_cutoff, beta, and d_cutoff
    # Update x_hat and y_hat with the smoothed values

    print("Smoothed Position:", x_hat, y_hat)