using Optimization, PyPlot, BenchmarkTools, LinearAlgebra, Random, Distributions
# structure to store optimization results
struct optimresult
	result::Vector{Float64}
	time::Float64
	iterations::Int64
	fevals::Int64
	fnval::Float64
end

# Function to define the information field
function infoField(pos::Vector{Float64})
	a = 3
	b = 5
	x, y = pos[1], pos[2]
	return (a - x)^2 + b * (y - x^2)^2
end
# cross-entropy optimizer
function crossentropy(f, x0, ϵ)
    μ = x0                              # Initial mean guess
    σ2 = Diagonal([1e3, 1e3])           # Initial covariance guess
    i = 1                               # Iterator
    elites = 10                         # Number of elite samples
    genpop = 100                        # Number of general samples
    maxi = 10000                        # Iterator upper bound

    # Use BenchmarkTools for precise timing
    timer = @elapsed begin
        # Optimization loop
        while (i < maxi && norm(σ2) > ϵ)
            σ2 +=  Diagonal(fill(1e-12, length(x0))); # ensure positive definite
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
function visualizeField2D(dims::Matrix{Int64})
	# Create the grid for evaluation
	xarray = collect(dims[1, 1]:0.01:dims[1, 2])  # Finer resolution
	yarray = collect(dims[2, 1]:0.01:dims[2, 2])

	# Generate grid points
	X, Y = generateMeshGrid(xarray, yarray)
	Z = [infoField([x, y]) for (x, y) in zip(X[:], Y[:])]

	# Plot the information field
	figure(figsize = (8, 6))
	contourf(X, Y, reshape(Z, size(X)), cmap = "viridis")
	colorbar(label = "Information Field Value")
	title("Information Field Visualization")
	xlabel("X")
	ylabel("Y")
	savefig("./figures/test.png", dpi = 300, bbox_inches = "tight")
	#show()
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

# Main script
begin
	noAgents::Int64 = 1             # Number of agents to consider
	dims = [-1 1; -1 1; 0 2]        # Environment dimensions
	visualizeField2D(dims)          # Create field visualization
	x0 = [0.0, 0.0]                 # Starting guess
	p = [1e3, 1e3]
    tol = 1e-2;
	# Iteratively solve position problem 
	for i in 1:noAgents
		resultStruct = crossentropy(reward,x0,tol)
        println(resultStruct.time)
	end
end
