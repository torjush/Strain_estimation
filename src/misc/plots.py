def plotGrid(ax, warped_grid, **kwargs):
    gridx = warped_grid[0, ::10, ::10]
    gridy = warped_grid[1, ::10, ::10]

    for i in range(gridx.shape[0] - 1, 1, -1):
        ax.plot(gridx[i, :], gridy[i, :], **kwargs)
    for i in range(gridx.shape[1] - 1, 1, -1):
        ax.plot(gridx[:, i], gridy[:, i], **kwargs)


def plotVectorField(ax, vector_field, **kwargs):
    U = vector_field[::-10, ::10, 0]
    V = vector_field[::-10, ::10, 1]

    ax.quiver(U, V, **kwargs)
