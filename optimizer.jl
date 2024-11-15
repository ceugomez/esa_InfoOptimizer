using Plots, Optimization

# Function to define the information field
function infoField(pos::Vector{Float64})
    a = 3
    b = 5
    x, y = pos[1], pos[2]
    return (a - x)^2 + b * (y - x^2)^2
end

# Function to visualize the information field
# Function to visualize the information field
function visualizeField(dims::Matrix{Int64})
    # Create the grid for evaluation
    xarray = dims[1, 1]:0.5:dims[1, 2]
    yarray = dims[2, 1]:0.5:dims[2, 2]

    # Initialize arrays for scatter plot data
    x_values = Float64[]
    y_values = Float64[]
    z_values = Float64[]

    for i in xarray
        for j in yarray
            push!(x_values, i)
            push!(y_values, j)
            push!(z_values, infoField([i, j]))
        end
    end

    # Normalize z_values to [0, 1] for consistent color mapping
    z_min, z_max = minimum(z_values), maximum(z_values)
    normalized_z = (z_values .- z_min) ./ (z_max - z_min)

    # Scatter plot in 3D with a colorscheme
    fig = scatter3d(
        x_values, y_values, z_values,
        color = normalized_z,               # Map normalized values to colors
        c = :viridis,                       # Use a colormap (e.g., Viridis)
        xlabel = "X", ylabel = "Y", zlabel = "Information Value",
        marker = :circle, legend = false,
        show=false
    )
    savefig(fig,"./figures/test.png")
    return fig;
end


# Main script
begin
    noAgents::Int64 = 3            # Number of agents to consider
    dims = [-1 1; -1 1; 0 2]  # Environment dimensions

    # Visualize the information field
    visualizeField(dims)
end
