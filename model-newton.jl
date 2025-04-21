using LinearAlgebra
using Plots

# === Podatki ===
# u = [2.0, 4.0, 5.0, -3.0]
# v = [-5.0, 0.0, 0.0, -5.0]
# x = [3.0, 0, -3.0, 0, 0, 3.0, 1.5]
# y = [0, -3.0, 0, 3.0, 2.4, -2.4, -2.5]
uReal = [4.0, 0, -4.0, 0]
vReal = [0, -4.0, 0, 4.0]
# Neznani položaji radarjev (u_k, v_k)
u = rand(-5:0.1:5, 4)
v = rand(-5:0.1:5, 4)
# Položaji opravljenih meritev (x_i, y_i)
x = rand(-8:0.1:8, 80)
y = rand(-8:0.1:8, 80)

# Funkcija V_i(u, v)
function signalStrength(xi, yi, u, v)
    return sum(1 / ((xi - ui)^2 + (yi - vi)^2) for (ui, vi) in zip(u, v))  
end

# Funkcija za izračun z_i
function strengths(x, y, uReal, vReal)
    return [signalStrength(xi, yi, uReal, vReal) for (xi, yi) in zip(x, y)]
end

# Parcialni odvodi
function DSignalStrengthU(xi, yi, ui, vi)
    return 2 * (xi - ui) / ((xi - ui)^2 + (yi - vi)^2)^2  
end

function DSignalStrengthV(xi, yi, ui, vi)
    return 2 * (yi - vi) / ((xi - ui)^2 + (yi - vi)^2)^2  
end

# === Newton's Method with improvements ===
function newton(z, x, y, x0; tol=1e-4, maxit=10000)
    koraki = []
    X = x0
    k = length(x0) ÷ 2
    J = zeros(length(x), length(x0))

    for n in 1:maxit
        u = X[1:k]
        v = X[k+1:end]

        r = [signalStrength(xi, yi, u, v) - zi for (xi, yi, zi) in zip(x, y, z)]

        for (j, (xi, yi)) in enumerate(zip(x, y))
            for l in 1:k
                J[j, l] = DSignalStrengthU(xi, yi, u[l], v[l])
                J[j, k + l] = DSignalStrengthV(xi, yi, u[l], v[l])
            end
        end

        λ = 1e-3  # Damping parameter
        dx = (J' * J + λ*I) \ (-J' * r)
        
        X_new = X + dx
        push!(koraki, X_new)

        if norm(dx) < tol
            break
        end

        X = X_new
    end

    return (X = X, koraki = koraki, n = length(koraki))
end

# Izračun jakosti z_i
z = strengths(x, y, uReal, vReal)
x0 = vcat(u, v) 
res_newton = newton(z, x, y, x0)

println("Initial guess: ", x0)
println("Final result: ", res_newton.X)
println("Norm of change: ", norm(res_newton.X - x0))
println("Number of steps: ", length(res_newton.koraki))

ures_newton = res_newton.X[1:end ÷ 2]
vres_newton = res_newton.X[end ÷ 2 + 1:end]

# === Visualization ===
xr = LinRange(-8, 8, 200)
yr = LinRange(-8, 8, 200)
zr = [log(log(signalStrength(xi, yi, uReal, vReal) + 1)) for xi in xr, yi in yr]
contour(xr, yr, zr; ratio = 1, colorbar=true, title="Radar Estimation (Improved)", xlabel="x", ylabel="y", size = (800, 600))

scatter!(x0[1:4], x0[5:8], label="Začetne ocene", color=:red, markersize=6)
scatter!(ures_newton, vres_newton, label="Ocene radarjev", color=:blue, markersize=8, marker=:diamond)
scatter!(uReal, vReal, marker=:star5, ms=5, label="Radarji", color=:green)
scatter!(x, y, label="Meritve", color=:yellow; markersize=5)

plot!(legend=:topleft, fontsize=5)
xlims!(-7, 7)
ylims!(-7, 7)

savefig("radar_newton_improved.png")
