using LinearAlgebra
using Plots

print("Radarji")

u = [1, 0, -1, 0]
v = [0, -1, 0, 1]

function signalStrength(x, y, u, v)
    return sum(1 / ((x - u[i])^2 + (y - v[i])^2) for i in eachindex(u))
end

signalStrength(1, 2, u, v)

