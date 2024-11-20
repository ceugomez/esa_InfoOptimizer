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
# Function to define the information field - Rastrigin
function infoField(pos::Vector{Float64})
    # Extract position components
    x, y = pos[1], pos[2]
    
    # Scaling factors
    xmin, xmax = -30.0, 30.0
    ymin, ymax = -30.0, 30.0
    
    # Normalize variables to [0, 1] range
    x_scaled = (x - xmin) / (xmax - xmin) * 5.12  # Scaling to match Rastrigin's typical domain
    y_scaled = (y - ymin) / (ymax - ymin) * 5.12
    
    # Rastrigin function parameters
    A = 10.0
    
    # Compute Rastrigin function
    info = 2 * A + (x_scaled^2 - A * cos(2 * π * x_scaled)) + (y_scaled^2 - A * cos(2 * π * y_scaled))
    
    return info
end

# Function to define the informatiion field - Ackley Mode
#=function infoField(pos::Vector{Float64})
    # Extract position components
    x, y = pos[1], pos[2]

    # Ackley parameters
    a = 20.0
    b = 0.2
    c = 2π

    # Compute Ackley function
    term1 = -a * exp(-b * sqrt(0.5 * (x^2 + y^2)))
    term2 = -exp(0.5 * (cos(c * x) + cos(c * y)))
    ackley = term1 + term2 + a + exp(1)

    return -ackley
end
=#
# Function to define the information field - Big Hill Mode
#= function infoField(pos::Vector{Float64})
    x, y, z = pos[1], pos[2], pos[3]
    R = sqrt((x / 3)^2 + (y / 3)^2)
    info = (-sin(R) + 1) * 30 - R^2# ensure positive
    return info
end
=#
# Function to define the information field - Rosenbrock Mode
#= function infoField(pos::Vector{Float64})
    # Extract position components
    x, y, z = pos[1], pos[2], pos[3]

    # Domain bounds
    xmin, xmax = -30.0, 30.0
    ymin, ymax = -30.0, 30.0
    zmin, zmax = 0.0, 2.0

    # Normalize variables to [0, 1] range
    x_scaled = (x - xmin) / ((xmax - xmin)*0.1)
    y_scaled = (y - ymin) / ((ymax - ymin)*0.1)
    z_scaled = (z - zmin) / ((zmax - zmin)*0.1)

    # Rosenbrock parameters
    a = 1.0
    b = 3.0
    c = 3.0

    # Scaled Rosenbrock function
    info = ((a - x_scaled)^2 + b * (y_scaled - x_scaled^2)^2)/1e3
    return info
end
 =#
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
    # convenience function - replaces MeshGrid from python
    function generateMeshGrid(xrange::AbstractVector, yrange::AbstractVector)
        X = repeat(xrange', length(yrange), 1)
        Y = repeat(yrange, 1, length(xrange))
        return X, Y
    end
    # Create the grid for evaluation
    xarray = collect(dims[1, 1]:0.1:dims[1, 2])
    yarray = collect(dims[2, 1]:0.1:dims[2, 2])

    # Generate grid points
    X, Y = generateMeshGrid(xarray, yarray)
    Z = [infoField([x, y, prob.pos[1][3]]) for (x, y) in zip(X[:], Y[:])]

    # Plot the information field
    for i in 1:length(flags)
        # Case: Contour Plot 
        if (flags[i] == true && i == 1)
            println("Plotting Contours....")
            plt = figure()
            ax = plt.add_subplot(111)
            contourf(X, Y, reshape(Z, size(X)), cmap="viridis", levels=collect(minimum(Z):((maximum(Z) - minimum(Z))/45):maximum(Z)))
            for j in 1:length(prob.pos)
                temp = ax.scatter(prob.pos[j][1], prob.pos[j][2], color="red", s=50, label="Point of Interest")
				temp.set_zorder(5)
            end
            colorbar(label="Information Field Value")
            title("Information Field Visualization")
            xlabel("X")
            ylabel("Y")
            savefig("./figures/contour.png", dpi=300, bbox_inches="tight")
        end
        # Case: Info Field Surface
        if (flags[i] == 1 && i == 2)
            println("Plotting Info Field Surface...")
            plt = figure()
            ax = plt.add_subplot(111, projection="3d")  # Create a 3D subplot
            surf = ax.plot_surface(X, Y, reshape(Z, size(X)), cmap="viridis", vmin=minimum(Z) * 2)  # Plot the surface
            # Scatter plot the point on the surface
            colorbar(surf)  # Add a colorbar for reference
            for j in 1:length(prob.pos)
				temp = ax.scatter(prob.pos[j][1], prob.pos[j][2], infoField(prob.pos[j]), color="red", s=150, label="Point of Interest")
				temp.set_zorder(5);
            end
            xlabel("X-axis")
            ylabel("Y-axis")
            ax.set_zlabel("Z-axis")
            title("Information Value Map |x=" * string(trunc(prob.pos[1][3], digits=2, base=10)))
            savefig("./figures/Info_surface.png", dpi=300, bbox_inches="tight")
        end
        # Case : Reward Function Surface
        if (flags[i] == 1 && i == 3)#show reward function for minimization
            println("Plotting Reward Field Surface....")
            Z = [reward([x, y, 1]) for (x, y) in zip(X[:], Y[:])]
            plt = figure()
            ax = plt.add_subplot(111, projection="3d")  # Create a 3D subplot
            surf = ax.plot_surface(X, Y, reshape(Z, size(X)), cmap="viridis", vmin=minimum(Z) * 2)  # Plot the surface
            Z = [infoField([x, y, 1]) for (x, y) in zip(X[:], Y[:])]
            colorbar(surf)  # Add a colorbar for reference
            for j in 1:length(prob.pos)
               	temp = ax.scatter(prob.pos[j][1], prob.pos[j][2], infoField(prob.pos[j]), color="red", s=150, label="Point of Interest")
				temp.set_zorder(5)
			end
			xlim(prob.dims[1,1],prob.dims[1,2]);
			ylim(prob.dims[2,1],prob.dims[2,2])
			zlim(minimum(Z)-5, maximum(Z)+5)

            xlabel("X-axis")
            ylabel("Y-axis")
            ax.set_zlabel("Z-axis")
            title("Reward Function Map Map |x=" * string(trunc(prob.pos[1][3], digits=2, base=10)))
            savefig("./figures/Reward_surface.png", dpi=300, bbox_inches="tight")
        end
    end
    show()
    return nothing
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

# Main script
begin
    global prob = problemStatement(5, [-30 30; -30 30; 0 2], "Cross-Entropy", Vector{Vector{Float64}}(),10) # Number of agents to consider, Environment dimension, opt method
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
    visualizeField(prob.dims, [true, true, true])          # Create field visualization
end
