using Optimization, PyPlot, BenchmarkTools, LinearAlgebra, Random, Distributions
# structure to store optimization results
struct optimresult
    result::Vector{Float64}
    time::Float64
    iterations::Int64
    fevals::Int64
    fnval::Float64
end
struct problemStatement
	agents::Int64
	dims::Matrix{Int64}
	OptMethod::String

end
# Function to define the information field
function infoField(pos::Vector{Float64})
    a = 3
    b = 5
	c = 12
    x, y, z = pos[1], pos[2], pos[3]
    return -((a - x)^2 + b * (y - x^2)^2 + c * (y - z^2))
end
# cross-entropy optimizer
function crossentropy(f, x0, ϵ)
    μ = x0                              # Initial mean guess
    σ2 = Diagonal([1e3, 1e3,1e3])           # Initial covariance guess
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
# convenience function - replaces MeshGrid from python
function generateMeshGrid(xrange::AbstractVector, yrange::AbstractVector)
    X = repeat(xrange', length(yrange), 1)
    Y = repeat(yrange, 1, length(xrange))
    return X, Y
end

# Function to visualize the information field
function visualizeField2D(dims::Matrix{Int64}, pos::Vector{Float64})
    # Create the grid for evaluation
    xarray = collect(dims[1, 1]:0.01:dims[1, 2])  
    yarray = collect(dims[2, 1]:0.01:dims[2, 2])  

    # Generate grid points
    X, Y = generateMeshGrid(xarray, yarray)
    Z = [infoField([x, y, 1]) for (x, y) in zip(X[:], Y[:])]

    # Plot the information field
    figure(figsize=(8, 6))
    contourf(X, Y, reshape(Z, size(X)), cmap="viridis")
	plot(pos[1],pos[2],pos[3]);
    colorbar(label="Information Field Value")
    title("Information Field Visualization")
    xlabel("X")
    ylabel("Y")
    savefig("./figures/test.png", dpi=300, bbox_inches="tight")
    show()
end


function reward(X)
    # Build base function
    value = infoField(X)
    if value == 0
        J = 1e12
    else
        J = 1 / value
    end
    return J
end
function printStartupScript(params::problemStatement)
	println("cgf cego6160@colorado.edu 11.18.24")
	println("Initializing...");
	println(string(params.agents)*" Agents")
	println("Dimensions: " *string(params.dims[1,2]-params.dims[1,1]) * " x " * string(params.dims[2,2]-params.dims[2,1]) * " x "*string(params.dims[3,2]-params.dims[3,1]))
	println("Optimization: " * params.OptMethod);
	println("Running...");
end
# Main script
begin
	prob = problemStatement(1, [-30 30; -30 30; 0 2],"Cross-Entropy" ) # Number of agents to consider, Environment dimension, opt method
	printStartupScript(prob)
    x0 = [0.0, 0.0, 0.0]               # Starting guess
    p = [1e3, 1e3, 1e3]
    tol = 1e-2;
    # Iteratively solve position problem 
    result = crossentropy(reward, x0, tol)
	visualizeField2D(prob.dims[1:2,1:2], result.result);          	# Create field visualization
end
