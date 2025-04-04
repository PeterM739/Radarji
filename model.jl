using LinearAlgebra
using Plots

u = [0.5, 0, -0.5, 0]
v = [0, -0.5, 0, 0.5]
x = [2.0, 0, -2.0, 0]
y = [0, -2.0, 0, 2.0]
xreal = [1.0, 0, -1.0, 0]
yreal = [0, -1.0, 0, 1.0]
z = [1.511, 1.511, 1.511, 1.511]

function signalStrength(xi, yi, u, v)
    return sum(1 / ((xi - u[i])^2 + (yi - v[i])^2) for i in eachindex(u))
end

function F(z, u, v, x, y) 
    return sum((signalStrength(x[i], y[i], u, v) - z[i])^2 for i in eachindex(z))
end

function DSignalStrengthU(xi, yi, ui, vi)
    return 2 * (xi - ui) / ((xi - ui)^2 + (yi - vi)^2)^2
end

function DSignalStrengthV(xi, yi, ui, vi)
    return 2 * (yi - vi) / ((xi - ui)^2 + (yi - vi)^2)^2
end

function GradF(z, u, v, x, y)
    du = [sum(2 * (signalStrength(xi, yi, u, v) - zi) * DSignalStrengthU(xi, yi, ui, vi) for (xi, yi, zi) in zip(x, y, z)) for (ui, vi) in zip(u, v)]
    dv = [sum(2 * (signalStrength(xi, yi, u, v) - zi) * DSignalStrengthV(xi, yi, ui, vi) for (xi, yi, zi) in zip(x, y, z)) for (ui, vi) in zip(u, v)]
    return vcat(du, dv)
end

function gradmet(z, x, y, alpha, x0; tol = 1e-6, maxit = 10000, record_steps = false)
    n = 1
    x1 = x0
    steps = [];
    if record_steps
        u0 = x1[1:length(x)]
        v0 = x1[length(x)+1:end]
        for (ui, vi) in zip(u0, v0)
            push!(steps, [ui; vi])
        end
    end
    for outer n in 1:maxit
        # one step of gradient descent
        u0 = x0[1:length(x)]
        v0 = x0[length(x)+1:end]
        x1 = x0 .- alpha .* GradF(z, u0, v0, x, y)
        # optionally add x to steps
        #
        if record_steps
            korakiu = x1[1:length(x)]
            korakiv = x1[length(x)+1:end]
            for (ui, vi) in zip(korakiu, korakiv)
                push!(steps, [ui; vi])
            end
        end
        #
        # check if new x is within tolerance and ...
        if norm(x1-x0) < alpha*tol # bolje, če pomnožimo z alpha
            break
        end
        # ... repeat id not
        x0 = x1
    end

    # a warning if maxit was reached
    if n == maxit
        @warn "no convergence after $maxit iterations"
    end
    # let's return a named tuple
    if record_steps
        return (X = x1, n = n, steps = steps)
    else
        return (X = x1, n = n)
    end
end

x0 = vcat(u, v)
X, _, koraki = gradmet(z, x, y, 0.01, x0; tol = 1e-6, record_steps = true)

xr = LinRange(-10, 10, 200)
yr = LinRange(-10, 10, 200)
z = [log(log(signalStrength(xi, yi, u, v) + 1)) for xi in xr, yi in yr]

contour(xr, yr, z; ratio = 1)
scatter!(x, y)
scatter!([Tuple(T) for T in koraki])
# plot!([Tuple(T) for T in koraki])

