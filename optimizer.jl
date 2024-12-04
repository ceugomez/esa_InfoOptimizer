using Optimization, PyPlot, BenchmarkTools, LinearAlgebra, Random, Distributions
# structure to store optimization results
struct optimresult
    result::Vector{Float64}
    time::Float64
    iterations::Int64
    fevals::Int64
    fnval::Float64
end
# global structure to store problem statement
struct problemStatement
    agents::Int64
    dims::Matrix{Int64}
    OptMethod::String
    pos::Vector{Vector{Float64}}
	R::Float64
end
# convenience function - print problem statement 
function printStartupScript(params::problemStatement)
    println("cgf cego6160@colorado.edu 11.18.24")
    println("Initializing...")
    println(string(params.agents) * " Agents")
    println("Dimensions: " * string(params.dims[1, 2] - params.dims[1, 1]) * " x " * string(params.dims[2, 2] - params.dims[2, 1]) * " x " * string(params.dims[3, 2] - params.dims[3, 1]))
    println("Optimization: " * params.OptMethod)
    println("Running...")
end
 # Function to define the information field - quadratic
function infoField(pos::Vector{Float64})
    x, y, z = pos[1], pos[2], pos[3]
    return -2(x+3)^2 - 3(y-5)^2 - 300*(z-1)^2 + 3000
end

# Modified reward with constraints 
function reward(X)
    # Extract position components
    x, y, z = X[1], X[2], X[3]

    # Compute base value from the information field
    value = infoField(X)

    # Define problem bounds
    xmin, xmax = prob.dims[1, 1], prob.dims[1, 2]
    ymin, ymax = prob.dims[2, 1], prob.dims[2, 2]
    zmin, zmax = prob.dims[3, 1], prob.dims[3, 2]

    function barrier(v, vmin, vmax)
        if v < vmin
            return (v - vmin)^2
        elseif v > vmax
            return (v - vmax)^2
        else
            return 0.0
        end
    end

    penalty = barrier(x, xmin, xmax) + barrier(y, ymin, ymax) + barrier(z, zmin, zmax)


    # Repulsion penalty for avoiding co-location with other agents
    repulsion_penalty = 0.0
    if !isempty(prob.pos)
        for other_agent in prob.pos
            distance_squared = (x - other_agent[1])^2 + (y - other_agent[2])^2 + (z - other_agent[3])^2
            if (distance_squared > 0 && distance_squared < (prob.R+5)^2)  # Avoid division by zero
                repulsion_penalty += 1 / distance_squared  # Higher penalty for closer agents
            end
        end
    end

    # Combine the base value, barrier function, and 
    J = -value + 10 * penalty + 20 * repulsion_penalty # Scale penalty to balance with objective

    return J
end
# cross-entropy optimizer
function crossentropy(f, x0, ϵ)
    μ = x0                              # Initial mean guess
    σ2 = Diagonal([1e3, 1e3, 1e3])           # Initial covariance guess
    i = 1                               # Iterator
    elites = 10                         # Number of elite samples
    genpop = 100                        # Number of general samples
    maxi = 10000                        # Iterator upper bound

    # Use BenchmarkTools for precise timing
    timer = @elapsed begin
        # Optimization loop
        while (i < maxi && norm(σ2) > ϵ)
            σ2 += Diagonal(fill(1e-12, length(x0))) # ensure positive definite
            dist = MvNormal(μ, σ2)       # Build distribution
            X = rand(dist, genpop)      # Get samples from distribution
            samples = Float64[]
            for j in 1:genpop
                push!(samples, f(X[:, j]))
            end
            p = sortperm(samples)       # Sort samples by function value
            elite = X[:, p[1:elites]]   # Select elite samples

            # Update mean and covariance
            μ = mean(elite, dims=2)[:]
            σ2 = cov(elite')
            i += 1
        end
    end

    # Return optimization result
    return optimresult(μ, timer, i, genpop * i, f(μ))
end
# Function to visualize the information field
function visualizeField(dims::Matrix{Int64}, flags::Vector{Bool})
    # Generate a mesh grid (replacement for Python's MeshGrid)
    function generateMeshGrid(xrange::AbstractVector, yrange::AbstractVector)
        X = repeat(xrange', length(yrange), 1)
        Y = repeat(yrange, 1, length(xrange))
        return X, Y
    end

    # Generate the grid for evaluation
    xarray = collect(dims[1, 1]:0.1:dims[1, 2])
    yarray = collect(dims[2, 1]:0.1:dims[2, 2])
    X, Y = generateMeshGrid(xarray, yarray)

    # Helper function to save plots with consistent settings
    function savePlot(fig, filename)
        savefig(fig, filename, dpi=300, bbox_inches="tight")
    end

    # Compute the information field values for the grid
    Z = [infoField([x, y, prob.pos[1][3]]) for (x, y) in zip(X[:], Y[:])]

    for i in 1:length(flags)
        if flags[i]
            plt = figure()

            if i == 1
                # Contour Plot
                println("Plotting Contours...")
                ax = plt.add_subplot(111)
                contourf(X, Y, reshape(Z, size(X)), cmap="viridis",
                         levels=collect(minimum(Z):((maximum(Z) - minimum(Z)) / 45):maximum(Z)))
                for j in 1:length(prob.pos)
                    ax.scatter(prob.pos[j][1], prob.pos[j][2], color="red", s=50, label="Point of Interest").set_zorder(5)
                end
                colorbar(label="Information Field Value")
                title("Information Field Visualization")
                xlabel("X")
                ylabel("Y")
                savePlot(plt, "./figures/contour.png")

            elseif i == 2
                # Info Field Surface
                println("Plotting Info Field Surface...")
                ax = plt.add_subplot(111, projection="3d")
                surf = ax.plot_surface(X, Y, reshape(Z, size(X)), cmap="viridis", vmin=minimum(Z) * 2)
                colorbar(surf)
                for j in 1:length(prob.pos)
                    ax.scatter(prob.pos[j][1], prob.pos[j][2], infoField(prob.pos[j]),
                               color="red", s=150, label="Point of Interest").set_zorder(5)
                end
                xlabel("X-axis")
                ylabel("Y-axis")
                ax.set_zlabel("Z-axis")
                title("Information Value Map | x = $(round(prob.pos[1][3], digits=2))")
                savePlot(plt, "./figures/Info_surface.png")

            elseif i == 3
                # Reward Function Surface
                println("Plotting Reward Field Surface...")
                Z_reward = [reward([x, y, 1]) for (x, y) in zip(X[:], Y[:])]
                ax = plt.add_subplot(111, projection="3d")
                surf = ax.plot_surface(X, Y, reshape(Z_reward, size(X)), cmap="viridis", vmin=minimum(Z_reward) * 2)
                colorbar(surf)
                for j in 1:length(prob.pos)
                    ax.scatter(prob.pos[j][1], prob.pos[j][2], infoField(prob.pos[j]),
                               color="red", s=150, label="Point of Interest").set_zorder(5)
                end
                xlim(prob.dims[1, 1], prob.dims[1, 2])
                ylim(prob.dims[2, 1], prob.dims[2, 2])
                zlim(minimum(Z_reward) - 5, maximum(Z_reward) + 5)
                xlabel("X-axis")
                ylabel("Y-axis")
                ax.set_zlabel("Z-axis")
                title("Reward Function Map | x = $(round(prob.pos[1][3], digits=2))")
                savePlot(plt, "./figures/Reward_surface.png")
            end
        end
    end

    show()
    return nothing
end
function plotInfoFieldContours(resolution::Float64)
    """
    Generate (x, y) and (y, z) contour plots of the information field using the global `prob` structure.

    Parameters:
        resolution::Float64 - Spacing between grid points (smaller values yield finer resolution).
    """
    global prob

    # Extract domain bounds from `prob.dims`
    xmin, xmax = prob.dims[1, :]
    ymin, ymax = prob.dims[2, :]
    zmin, zmax = prob.dims[3, :]

    # Generate grid points
    x_vals = collect(xmin:resolution:xmax)
    y_vals = collect(ymin:resolution:ymax)
    z_vals = collect(zmin:resolution:zmax)

    # Compute (x, y) contour
    X, Y = repeat(x_vals', length(y_vals), 1), repeat(y_vals, 1, length(x_vals))
    Z_xy = [infoField([x, y, 1.0]) for (x, y) in zip(X[:], Y[:])]

    # Plot (x, y) contour
    plt.figure()
    contourf(X, Y, reshape(Z_xy, size(X)), cmap="magma", levels=50)
    colorbar(label="Info Field Value")
    title("Information Field Contour: (x, y)")
    xlabel("x")
    ylabel("y")
    # Overlay agent positions (x, y)
    for (i,agent) in enumerate(prob.pos)
        scatter(agent[1], agent[2], color="red", label="Agent $i", s=50)
    end
    legend()
    savefig("./figures/xy_contour.png", dpi=300, bbox_inches="tight")
    println("Saved (x, y) contour plot to './figures/xy_contour.png'.")

    # Compute (y, z) contour
    Y, Z = repeat(y_vals', length(z_vals), 1), repeat(z_vals, 1, length(y_vals))
    Z_yz = [infoField([0.0, y, z]) for (y, z) in zip(Y[:], Z[:])]

    # Plot (y, z) contour
    plt.figure()
    contourf(Y, Z, reshape(Z_yz, size(Y)), cmap="magma", levels=50)
    colorbar(label="Info Field Value")
    title("Information Field Contour: (y, z)")
    xlabel("y")
    ylabel("z")
    # Overlay agent positions (y, z)
    for (i,agent) in enumerate(prob.pos)
        scatter(agent[2], agent[3], color="red", label="Agent $i", s=50)
    end
    legend()
    savefig("./figures/yz_contour.png", dpi=300, bbox_inches="tight")
    println("Saved (y, z) contour plot to './figures/yz_contour.png'.")
    show()
    return nothing
end

function plotInfoFieldLineContours(resolution::Float64)
    """
    Generate (x, y), (y, z), and (x, z) contour plots of the information field using the global `prob` structure,
    and overlay agent positions on the contours.

    Parameters:
        resolution::Float64 - Spacing between grid points (smaller values yield finer resolution).
    """
    global prob

    # Extract domain bounds from `prob.dims`
    xmin, xmax = prob.dims[1, :]
    ymin, ymax = prob.dims[2, :]
    zmin, zmax = prob.dims[3, :]

    # Generate grid points
    x_vals = collect(xmin:resolution:xmax)
    y_vals = collect(ymin:resolution:ymax)
    z_vals = collect(zmin:resolution:zmax)

    # Compute (x, y) contour
    X, Y = repeat(x_vals', length(y_vals), 1), repeat(y_vals, 1, length(x_vals))
    Z_xy = [infoField([x, y, 1.0]) for (x, y) in zip(X[:], Y[:])]

    # Plot (x, y) contour
    plt.figure()
    contour_lines = plt.contour(X, Y, reshape(Z_xy, size(X)), cmap="viridis", levels=20)
    plt.clabel(contour_lines, inline=1, fontsize=8)
    plt.title("Information Field Contour: (x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    # Overlay agent positions (x, y)
    for (i, agent) in enumerate(prob.pos)
        scatter(agent[1], agent[2], color="red", label="Agent $i", s=50)
    end
    plt.legend()
    savefig("./figures/xy_contour_lines.png", dpi=300, bbox_inches="tight")
    println("Saved (x, y) contour plot to './figures/xy_contour_lines.png'.")

    # Compute (y, z) contour
    Y, Z = repeat(y_vals', length(z_vals), 1), repeat(z_vals, 1, length(y_vals))
    Z_yz = [infoField([0.0, y, z]) for (y, z) in zip(Y[:], Z[:])]

    # Plot (y, z) contour
    plt.figure()
    contour_lines = plt.contour(Y, Z, reshape(Z_yz, size(Y)), cmap="viridis", levels=20)
    plt.clabel(contour_lines, inline=1, fontsize=8)
    plt.title("Information Field Contour: (y, z)")
    plt.xlabel("y")
    plt.ylabel("z")
    # Overlay agent positions (y, z)
    for (i, agent) in enumerate(prob.pos)
        scatter(agent[2], agent[3], color="red", label="Agent $i", s=50)
    end
    plt.legend()
    savefig("./figures/yz_contour_lines.png", dpi=300, bbox_inches="tight")
    println("Saved (y, z) contour plot to './figures/yz_contour_lines.png'.")

    # Compute (x, z) contour
    X, Z = repeat(x_vals', length(z_vals), 1), repeat(z_vals, 1, length(x_vals))
    Z_xz = [infoField([x, 0.0, z]) for (x, z) in zip(X[:], Z[:])]

    # Plot (x, z) contour
    plt.figure()
    contour_lines = plt.contour(X, Z, reshape(Z_xz, size(X)), cmap="viridis", levels=20)
    plt.clabel(contour_lines, inline=1, fontsize=8)
    plt.title("Information Field Contour: (x, z)")
    plt.xlabel("x")
    plt.ylabel("z")
    # Overlay agent positions (x, z)
    for (i, agent) in enumerate(prob.pos)
        scatter(agent[1], agent[3], color="red", label="Agent $i", s=50)
    end
    plt.legend()
    savefig("./figures/xz_contour_lines.png", dpi=300, bbox_inches="tight")
    println("Saved (x, z) contour plot to './figures/xz_contour_lines.png'.")

    plt.show()
    return nothing
end




# Main script
begin
    global prob = problemStatement(1, [-30 30; -30 30; 0 2], "Cross-Entropy", Vector{Vector{Float64}}(),10) # Number of agents to consider, Environment dimension, opt method
    printStartupScript(prob)
    x0 = [0.0, 0.0, 0.0]               # Starting guess
    p = [60, 60, 2]
    tol = 1e-12
    # Iteratively solve position problem 
    for i in 1:prob.agents
        # optimize
        result = crossentropy(reward, x0, tol)
        # output position
        println("Agent Position: " * string(trunc(result.result[1], digits=3, base=10)) * ", " * string(trunc(result.result[2], digits=3, base=10)) * ", " * string(trunc(result.result[3], digits=3, base=10)))
        push!(prob.pos, result.result)
    end

    #visualizeField(prob.dims, [true, true, true])          # Create field visualization
    plotInfoFieldContours(0.5)  
    plotInfoFieldLineContours(0.1)
end
