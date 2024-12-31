import numpy as np

def ODELearning(trace, num_var, num_ud):

    len_labels = len(trace[0]['labels_num'])  # Number of clusters
    ode = []

    for label in range(1, len_labels + 1):
        x_seg = []  # List to store state segments
        ud_seg = []  # List to store control input segments

        x_list = []
        ud_list = []
        x_plus_list = []

        # Extract all segments from all traces associated with current cluster label
        for j in range(len(trace)):
            labels_trace = trace[j]['labels_trace']  # Cluster labels for the current trace
            idx = np.where(labels_trace == label)[0]  # Find the indices of the current label in labels_trace
            x = trace[j]['x']
            ud = trace[j]['ud'] if num_ud else np.array([])  # Control inputs

            # Get the start and end indices of the segments for this label
            startj = []
            endj = []

            for ii in idx:
                startj.append(trace[j]['chpoints'][ii] - 1)
                endj.append(trace[j]['chpoints'][ii + 1] - 2)

            # Loop over each segment associated with the current label

            for n in range(len(startj)):
                x_list.append(x[:, startj[n]:endj[n]])

                if num_ud:
                    ud_list.append(ud[:, startj[n]:endj[n]])
                else:
                    ud_list.append(np.zeros((num_ud, endj[n] - startj[n])) ) # Placeholder for empty control input vectors
                # Extract state space vector for the segment

                x_plus_list.append(x[:, startj[n] + 1 :endj[n] + 1])

        # Convert lists to numpy arrays
        x_seg = np.hstack(x_list)  # Flatten and concatenate the state space vectors
        ud_seg = np.hstack(ud_list)  # Flatten and concatenate the control input vectors
        x_seg_plus = np.hstack(x_plus_list)

        # Solve the ODE model for the current cluster (A and B matrices)
        # A and B will be computed using matrix division (equivalent to the left matrix division in MATLAB)
        MM = np.linalg.lstsq(np.vstack([x_seg, ud_seg]).T, x_seg_plus.T, rcond=None)[0]


    return ode