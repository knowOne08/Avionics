
import numpy as np
import csv
import matplotlib.pyplot as plt

def rocket_trajectory(initial_position, v0, firing_angle):
    # Constants
    g = 9.801
    angle_rad = np.deg2rad(firing_angle)
    theta = angle_rad    
    x0 = initial_position[0]
    z0 = initial_position[1]
    time = 0.0

    # Raw data values arrays
    vx = []
    vz = []
    px = []
    pz = []
    flight_path_angle = []

    apogee_index = None
    end_of_flight = 0
    n = 0
    # Velocity data store
    while True:
        vx.append(v0 * np.cos(theta))
        vz.append(-(v0 * np.sin(theta) - g*(n/100)))
        
        flight_path_angle.append(np.arctan((vz[n]/vx[n])))
        
        if apogee_index is None or abs(vz[n]) < abs(vz[apogee_index]):
            apogee_index = n
            
        if np.sqrt((vz[n]*vz[n]) + (vx[n]*vx[n])) > abs(v0) and n > 1:
            end_of_flight = n
            break  # Rocket hits the ground
        n+=1
        
    n = 1
    xz = 0 
    # Position data store
    while n != end_of_flight:
        px.append(x0 + v0 * np.cos(theta)*(n/100))

        xz += (vz[n-1] + vz[n])*0.01/2
        
        pz.append(xz)
        if pz[n-1] > 0:
            break

        n+=1

    # print(pz)
    apogee = pz[apogee_index]

    # Calculate final velocity
    final_velocity = np.sqrt((vz[end_of_flight]*vz[end_of_flight]) + (vx[end_of_flight]*vx[end_of_flight]))

    return final_velocity, np.array(px), np.array(pz), np.array(vx), np.array(vz), flight_path_angle, n*0.01, apogee, apogee_index

def export_to_csv(final_velocity, px, pz, vx, vz, flight_path_angle, total_time, apogee_altitude, filename):
    # Convert single values to arrays if necessary
    final_velocity = np.atleast_1d(final_velocity)
    px = np.atleast_1d(px)
    pz = np.atleast_1d(pz)
    vx = np.atleast_1d(vx)
    vz = np.atleast_1d(vz)

    # Determine maximum length among arrays
    max_length = max(len(final_velocity), len(px), len(pz), len(vx), len(vz))
    # Pad arrays with placeholder values if necessary
    final_velocity = np.pad(final_velocity, (0, max_length - len(final_velocity)), mode='constant', constant_values=np.nan)
    px = np.pad(px, (0, max_length - len(px)), mode='constant', constant_values=np.nan)
    pz = np.pad(pz, (0, max_length - len(pz)), mode='constant', constant_values=np.nan)
    vx = np.pad(vx, (0, max_length - len(vx)), mode='constant', constant_values=np.nan)
    vz = np.pad(vz, (0, max_length - len(vz)), mode='constant', constant_values=np.nan)

    time_index = np.arange(max_length) * total_time / max_length

    # Combine data into rows
    rows = zip(time_index,final_velocity, px, pz, vx, vz, flight_path_angle, [total_time] * max_length, [apogee_altitude] * max_length)

    # Write data to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['time','final_velocity', 'px', 'pz', 'vx', 'vz', 'flight_path_angle', 'total_time', 'apogee_altitude'])
        
        # Write rows
        for row in rows:
            writer.writerow(row)



def plot_trajectory(px, pz):
    # Flip pz values to make y-axis point upward
    # Plotting
    plt.plot(px, pz)

    # Invert y-axis
    plt.gca().invert_yaxis()

    # Optionally, you can set the y-axis ticks to reflect the original negative values
    # plt.yticks(np.arange(min(pz), max(pz), 5))
    plt.ylabel('Altitude (m)')
    plt.xlabel('Range (m)')
    plt.title('Rocket Trajectory')
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()

# Input parameters
initial_position = [0,0,0]
for i in range(3):
    initial_position.append(float(input(f"Enter initial position for {'xyz'[i]}: ")))

# Get user input for initial velocity
v0 = float(input("Enter initial velocity (m/s): "))

# Get user input for firing angle
firing_angle = float(input("Enter firing angle (degrees): "))

# Simulate rocket trajectory
final_velocity, px, pz, vx, vz, flight_path_angle, total_time, apogee_altitude, apogee_index = rocket_trajectory(initial_position, v0, firing_angle)

# Print results
print("Apogee: ", apogee_altitude)
print("Apogee Velocity: ", vx[apogee_index])
print("Apogee time: ", apogee_index*0.01)
print("Final position: [",px[-1], ",0,", pz[-1], "]")
print("Final velocity:", final_velocity)
print("Final flight path angle (degrees):", np.rad2deg(flight_path_angle[-1]))
print("Final flight time:", total_time)

# Plot trajectory and export data
filename = 'data.csv'
export_to_csv(final_velocity, px, pz, vx, vz, flight_path_angle, total_time, apogee_altitude, filename)
plot_trajectory(px,pz)

