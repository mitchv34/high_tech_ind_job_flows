## Configure Plots
theme(:vibrant) # Color theme
default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style



using NLsolve

# Define the function
f(x) = (a .* 2 .^ (1 ./ x) .+ c)./(1 .+ 2 .^(1 ./ x)) .- m

# Define the parameters
m = 35
a = 2176594
c = 0

# Find the root of the function
sol = nlsolve(f, [ -0.06], ftol = 1e-16)
# Print the root
println("The root of the function is: ", sol.zero[1])

