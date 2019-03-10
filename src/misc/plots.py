def plotGrid(ax, warped_grid, **kwargs):
    gridx = warped_grid[7, 0, ::10, ::10]
    gridy = warped_grid[7, 1, ::10, ::10]

    for i in range(gridx.shape[0]):
        ax.plot(gridx[i, :], gridy[i, :], **kwargs)
    for i in range(gridx.shape[1]):
        ax.plot(gridx[:, i], gridy[:, i], **kwargs)


def plotVectorField(ax, vector_field, **kwargs):
    U = vector_field.numpy()[7, :, :, 0]
    V = vector_field.numpy()[7, :, :, 1]

    ax.quiver(U, V, **kwargs)
