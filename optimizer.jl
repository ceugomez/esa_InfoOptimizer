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

end
# Function to define the information field
function infoField(pos::Vector{Float64})
    x, y, z = pos[1], pos[2], pos[3]
    R = sqrt((x/3)^2 + (y/3)^2)
	info = (-sin(R)+1)*30 - R^2	# ensure positive
	return info;
end
# reward function - contains barrier function constraints
#function reward(X)
#    # Build base function
#   value = infoField(X)
#	# add constraints#
#	x,y,z = X[1],X[2],X[3];
#	if (x<prob.dims[1,1] || x>prob.dims[1,2] || y<prob.dims[2,1] || y > prob.dims[2,2] || z < prob.dims[3,1] || z > prob.dims[3,2])
#		J = 10
#	else
#		J = -value
#	end
 #   return J
#end
function reward(X)
    # Extract position components
    x, y, z = X[1], X[2], X[3]
    
    # Compute base value from the information field
    value = infoField(X)
    
    # Define problem bounds
    xmin, xmax = prob.dims[1, 1], prob.dims[1, 2]
    ymin, ymax = prob.dims[2, 1], prob.dims[2, 2]
    zmin, zmax = prob.dims[3, 1], prob.dims[3, 2]
    
    # Smooth barrier function for constraints
    function barrier(v, vmin, vmax)
        if v < vmin
            return (v - vmin)^2
        elseif v > vmax
            return (v - vmax)^2
        else
            return 0.0
        end
    end
    
    # Apply the barrier to each dimension
    penalty = barrier(x, xmin, xmax) + barrier(y, ymin, ymax) + barrier(z, zmin, zmax)
    
    # Combine the base value and penalty
    J = -value + 10 * penalty  # Scale penalty to balance with objective
    
    return J
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
# Function to visualize the information field
function visualizeField(dims::Matrix{Int64}, pos::Vector{Float64}, flags::Vector{Bool})
    # Create the grid for evaluation
    xarray = collect(dims[1, 1]:0.01:dims[1, 2])  
    yarray = collect(dims[2, 1]:0.01:dims[2, 2])  

    # Generate grid points
    X, Y = generateMeshGrid(xarray, yarray)
    Z = [infoField([x, y, 1]) for (x, y) in zip(X[:], Y[:])]

    # Plot the information field
	for i in 1:length(flags)
		# Case: Contour Plot 
		if (flags[i]==true && i==1)
			println("Plotting Contours....");
			plt = figure()
			ax = plt.add_subplot(111);
			contourf(X, Y, reshape(Z, size(X)), cmap="viridis", levels=collect(minimum(Z):1:maximum(Z)))
			ax.scatter(pos[1], pos[2], color="red", s=50, label="Point of Interest");
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
			ax.scatter(pos[1], pos[2], infoField(pos), color="red", s=50, label="Point of Interest")
			xlabel("X-axis")
			ylabel("Y-axis")
			ax.set_zlabel("Z-axis") 
			title("Information Value Map |x="*string(trunc(pos[3],digits=2, base=10)));
			savefig("./figures/Info_surface.png", dpi=300, bbox_inches="tight")
			show()
		end
		# Case : Reward Function Surface
		if (flags[i] == 1 && i==3)	#show reward function for minimization
			println("Plotting Reward Field Surface....")
			Z = [reward([x, y, 1]) for (x, y) in zip(X[:], Y[:])]
			plt = figure()
			ax = plt.add_subplot(111, projection="3d")  # Create a 3D subplot
			surf = ax.plot_surface(X, Y, reshape(Z, size(X)), cmap="viridis", vmin=minimum(Z) * 2)  # Plot the surface
			Z = [infoField([x, y, 1]) for (x, y) in zip(X[:], Y[:])]
			colorbar(surf)  # Add a colorbar for reference
			ax.scatter(pos[1], pos[2], reward(pos), color="red", s=50, label="Point of Interest")
			xlabel("X-axis")
			ylabel("Y-axis")
			ax.set_zlabel("Z-axis") 
			title("Reward Function Map Map |x="*string(trunc(pos[3],digits=2, base=10)));
			savefig("./figures/Reward_surface.png", dpi=300, bbox_inches="tight")
			show()
		end
	end
	return nothing
end
# convenience function - replaces MeshGrid from python
function generateMeshGrid(xrange::AbstractVector, yrange::AbstractVector)
    X = repeat(xrange', length(yrange), 1)
    Y = repeat(yrange, 1, length(xrange))
    return X, Y
end
# convenience function - print problem statement 
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
	global prob = problemStatement(1, [-30 30; -30 30; 0 2],"Cross-Entropy" ) # Number of agents to consider, Environment dimension, opt method
	printStartupScript(prob)
    x0 = [0.0, 0.0, 0.0]               # Starting guess
    p = [60, 60, 2]
    tol = 1e-5;
    # Iteratively solve position problem 
    result = crossentropy(reward, x0, tol)
	println("Agent Position: "*string(trunc(result.result[1], digits=3, base=10))*", "*string(trunc(result.result[2], digits=3, base=10))*", "*string(trunc(result.result[3], digits=3, base=10)));
	visualizeField(prob.dims, result.result, [true,true,true]);          	# Create field visualization

	# set up default solver
	#prob = OptimizationProblem(reward, x0, p);
	#sol = solve(prob);
	#print(sol);
end
