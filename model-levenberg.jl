using LinearAlgebra
using Plots

# === Real radar positions ===
uReal = [4.0, 0.0, -4.0, 0.0]
vReal = [0.0, -4.0, 0.0, 4.0]

# === Random initial guess for radar positions ===
u = rand(-5:0.1:5, 4)
v = rand(-5:0.1:5, 4)

# === Random measurement positions ===
x = rand(-8:0.1:8, 80)
y = rand(-8:0.1:8, 80)

# === Signal model ===
function signalStrength(xi, yi, u, v)
    sum(1 / ((xi - ui)^2 + (yi - vi)^2) for (ui, vi) in zip(u, v))
end

function strengths(x, y, uReal, vReal)
    [signalStrength(xi, yi, uReal, vReal) for (xi, yi) in zip(x, y)]
end

function DSignalStrengthU(xi, yi, ui, vi)
    2 * (xi - ui) / ((xi - ui)^2 + (yi - vi)^2)^2
end

function DSignalStrengthV(xi, yi, ui, vi)
    2 * (yi - vi) / ((xi - ui)^2 + (yi - vi)^2)^2
end

function jacobian(x, y, X)
    k = length(X) ÷ 2
    u = X[1:k]
    v = X[k+1:end]
    J = zeros(length(x), length(X))

    for (j, (xi, yi)) in enumerate(zip(x, y))
        for l in 1:k
            J[j, l] = DSignalStrengthU(xi, yi, u[l], v[l])
            J[j, k + l] = DSignalStrengthV(xi, yi, u[l], v[l])
        end
    end

    return J
end

# === Improved Newton's Method (Levenberg–Marquardt style) ===
function newton(z, x, y, x0; tol=1e-4, maxit=10000)
    koraki = [x0]
    X = x0
    k = length(X) ÷ 2
    λ = 10.0

    for n in 1:maxit
        u = X[1:k]
        v = X[k+1:end]
        r = [signalStrength(xi, yi, u, v) - zi for (xi, yi, zi) in zip(x, y, z)]
        J = jacobian(x, y, X)

        dx = (J' * J + λ * I(length(X))) \ (-J' * r)
        X_new = X + dx

        u_new = X_new[1:k]
        v_new = X_new[k+1:end]
        r_new = [signalStrength(xi, yi, u_new, v_new) - zi for (xi, yi, zi) in zip(x, y, z)]

        if norm(r_new) < norm(r)
            λ /= 10
            X = X_new
            push!(koraki, X)
        else
            λ *= 4
        end

        if norm(dx) < tol
            break
        end
    end

    return (X = X, koraki = koraki, n = length(koraki))
end

function levenberg(z, x, y, x0; tol=1e-4, maxit=10000)
    X = x0
    k = length(X) ÷ 2
    λ = 10.0
    koraki = [x0]  # Added this line

    for n in 1:maxit
        u = X[1:k]
        v = X[k+1:end]
        r = [signalStrength(xi, yi, u, v) - zi for (xi, yi, zi) in zip(x, y, z)]
        J = jacobian(x, y, X)

        dx = (J' * J + λ * Matrix{Float64}(I, length(X), length(X))) \ (-J' * r)
        X_new = X + dx

        u_new = X_new[1:k]
        v_new = X_new[k+1:end]
        r_new = [signalStrength(xi, yi, u_new, v_new) - zi for (xi, yi, zi) in zip(x, y, z)]

        if norm(r_new) < norm(r)
            λ /= 10
            X = X_new
            push!(koraki, X)
        else
            λ *= 4
        end

        if norm(dx) < tol
            break
        end
    end

    return (X = X, koraki = koraki, n = length(koraki))  # Return koraki too for consistency
end

# === Execute ===
z = strengths(x, y, uReal, vReal)
x0 = vcat(u, v)
res_newton = levenberg(z, x, y, x0)

println("Initial guess: ", x0)
println("Final result: ", res_newton.X)
println("Norm of change: ", norm(res_newton.X - x0))
println("Number of steps: ", res_newton.n)

ures_newton = res_newton.X[1:end ÷ 2]
vres_newton = res_newton.X[end ÷ 2 + 1:end]

# === Visualization ===
xr = LinRange(-8, 8, 200)
yr = LinRange(-8, 8, 200)
zr = [log(log(signalStrength(xi, yi, uReal, vReal) + 1)) for xi in xr, yi in yr]

contour(xr, yr, zr; ratio = 1, colorbar=true, title="Radar Estimation Levenberg",
        xlabel="x", ylabel="y", size = (800, 600))

scatter!(x0[1:4], x0[5:8], label="Začetne ocene", color=:red, markersize=6)
scatter!(ures_newton, vres_newton, label="Ocene radarjev", color=:blue, markersize=8, marker=:diamond)
scatter!(uReal, vReal, marker=:star5, ms=5, label="Radarji", color=:green)
scatter!(x, y, label="Meritve", color=:yellow, markersize=5)

plot!(legend=:topleft, fontsize=5)
xlims!(-7, 7)
ylims!(-7, 7)

savefig("radar_newton_improved.png")
