using Optimization, PlotlyJS

# Function to define the information field
function infoField(pos::Vector{Float64})
    a = 3
    b = 5
    x, y = pos[1], pos[2]
    return (a - x)^2 + b * (y - x^2)^2
end

# Function to visualize the information field
function visualizeField(dims::Matrix{Int64})
    # Create the grid for evaluation
    xarray = dims[1, 1]:0.1:dims[1, 2]
    yarray = dims[2, 1]:0.1:dims[2, 2]

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

    fig = plot(scatter(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="markers",
        marker=attr(
            size=3,
            color=z_values,                # set color to an array/list of desired values
            colorscale="Viridis",   # choose a colorscale
            opacity=0.8
        ),
        dpi=1200,
        type="scatter3d"
        ), Layout(margin=attr(l=0, r=0, b=0, t=0)))
    savefig(fig,"./figures/test.png")
    return fig;
end
function reward(X, agentPositions)
    # build base function 
    J = infoField(X);


end

# Main script
begin
    noAgents::Int64 = 3             # no. of agents to consider
    dims = [-1 1; -1 1; 0 2]        # environment dimensions
    visualizeField(dims)            # create field visualization
    x0 = [0 0];                     # starting guess
    p = [1e3, 1e3];
    for i in noAgents
        problem = OptimizationProblem(reward, x0, p)        
    end
end
