import matplotlib.pyplot as plt
from io import StringIO

def plot_optimizer_convergence_data(file_name: str):
    """
    Plot the convergence data of the optimization across iterates.

    :param file_name: name of the IPOPT output file
    :type file_name: str

    :return: None
    """

    # Read the IPOPT output file line by line
    myfile = open(file_name, 'rt')
    data = 'iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls\n'

    count = 0
    while True:
        # Get next line from file
        line = myfile.readline()

        # if line is empty: end of file is reached
        if not line:
            break

        # Look for iteration number
        if str(count) in line[:4]:
            count = count + 1
            # Remove r from the iteration line
            for tag in ['f', 'F', 'h', 'H', 'k', 'K', 'n', 'N', 'R', 'w', 's', 't', 'T', 'r']:
                if tag in line:
                    line = line.replace(tag, '')

            # ADd line to data file
            data = data + line

    # Close the file
    myfile.close()

    # Write the file in csv format and convert to pandas data frame
    data = StringIO(data)
    data = pd.read_csv(data, delim_whitespace=True)

    # Plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    font_size = 14

    ax[0, 0].plot(data['iter'].values, data['objective'].values)
    ax[0, 0].set_xlabel('Iterations', fontsize=font_size)
    ax[0, 0].set_ylabel('Objective', fontsize=font_size)
    ax[0, 0].tick_params(axis='both', labelsize=font_size)
    ax[0, 0].grid()

    ax[0, 1].semilogy(data['iter'].values, data['inf_pr'], label='inf_pr')
    ax[0, 1].semilogy(data['iter'].values, data['inf_du'], label='inf_du')
    ax[0, 1].set_xlabel('Iterations', fontsize=font_size)
    ax[0, 1].set_ylabel('Infeasibility', fontsize=font_size)
    ax[0, 1].tick_params(axis='both', labelsize=font_size)
    ax[0, 1].grid()
    ax[0, 1].legend(fontsize=font_size, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3, borderaxespad=0, frameon=False)

    ax[0, 2].plot(data['iter'].values, data['lg(mu)'], label='$log(\mu)$')
    ax[0, 2].set_xlabel('Iterations', fontsize=font_size)
    ax[0, 2].set_ylabel('Barrier parameter', fontsize=font_size)
    ax[0, 2].tick_params(axis='both', labelsize=font_size)
    ax[0, 2].grid()
    ax[0, 2].legend(fontsize=font_size, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3, borderaxespad=0, frameon=False)

    ax[1, 0].semilogy(data['iter'].values, data['||d||'].values)
    ax[1, 0].set_xlabel('Iterations', fontsize=font_size)
    ax[1, 0].set_ylabel('||d||', fontsize=font_size)
    ax[1, 0].tick_params(axis='both', labelsize=font_size)
    ax[1, 0].grid()
    ax[1, 0].set_ylim(1e-10, 1e3)

    ax[1, 1].plot(data['iter'].values, data['alpha_pr'].values, label='alpha_pr')
    ax[1, 1].plot(data['iter'].values, data['alpha_du'].values, label='alpha_du')
    ax[1, 1].legend(fontsize=font_size, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3, borderaxespad=0,
                    frameon=False)
    ax[1, 1].set_xlabel('Iterations', fontsize=font_size)
    ax[1, 1].set_ylabel('Stepsize', fontsize=font_size)
    ax[1, 1].tick_params(axis='both', labelsize=font_size)
    ax[1, 1].grid()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    fig.delaxes(ax[1, 2])

    return None
