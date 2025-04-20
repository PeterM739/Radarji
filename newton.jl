using LinearAlgebra
using Plots

# === Data ===
u = [2.0, 0.0, 5.0, -3.0]
v = [-3.0, 3.5, 0.0, -3.0]
x = [3.0, 0, -3.0, 0, 0, 0, 3.0, 1.5]
y = [0, -3.0, 0, 3.0, 2.4, -2.4, -2.5, 1.5]
uReal = [4.0, 0, -4.0, 0]
vReal = [0, -4.0, 0, 4.0]

# === Signal strength functions (modified for numerical stability) ===
function signalStrength(xi, yi, u, v)
    return sum(1 / ((xi - ui)^2 + (yi - vi)^2 + 1e-2) for (ui, vi) in zip(u, v))  # Added small constant
end

function strengthes(x, y, uReal, vReal)
    return [signalStrength(xi, yi, uReal, vReal) for (xi, yi) in zip(x, y)]
end

function DSignalStrengthU(xi, yi, ui, vi)
    return 2 * (xi - ui) / ((xi - ui)^2 + (yi - vi)^2 + 1e-2)^2  # Same constant
end

function DSignalStrengthV(xi, yi, ui, vi)
    return 2 * (yi - vi) / ((xi - ui)^2 + (yi - vi)^2 + 1e-2)^2  # Same constant
end

# === Newton's Method with improvements ===
function newton(z, x, y, x0; tol=1e-4, maxit=1000, λ_reg=0.01)
    koraki = [x0]
    X = x0
    k = length(x0) ÷ 2

    for i in 1:maxit
        u = X[1:k]
        v = X[k+1:end]

        r = [signalStrength(xi, yi, u, v) - zi for (xi, yi, zi) in zip(x, y, z)]
        J = zeros(length(x), 2k)

        for (j, (xi, yi)) in enumerate(zip(x, y))
            for l in 1:k
                J[j, l] = DSignalStrengthU(xi, yi, u[l], v[l])
                J[j, k + l] = DSignalStrengthV(xi, yi, u[l], v[l])
            end
        end

        λ = 1e-3  # Damping parameter
        # Add regularization to keep radars near initial guess:
        dx = (J' * J + λ*I + λ_reg*I) \ (-J' * r + λ_reg*(x0 - X))
        
        X_new = X + dx
        # Apply physical constraints (keep within bounds)
        X_new = max.(min.(X_new, [7,7,7,7,7,7,7,7]), [-7,-7,-7,-7,-7,-7,-7,-7])
        push!(koraki, X_new)

        if norm(dx) < tol
            break
        end

        X = X_new
    end

    return (X = X, koraki = koraki)
end

# === Run with better initial guess ===
z = strengthes(x, y, uReal, vReal)
x0 = vcat(u, v)  # Start near true solution
res_newton = newton(z, x, y, x0, λ_reg=0.1)

println("Initial guess: ", x0)
println("Final result: ", res_newton.X)
println("Norm of change: ", norm(res_newton.X - x0))
println("Number of steps: ", length(res_newton.koraki))

ures_newton = res_newton.X[1:end ÷ 2]
vres_newton = res_newton.X[end ÷ 2 + 1:end]

# === Enhanced Plot ===
xr = LinRange(-7, 7, 200)
yr = LinRange(-7, 7, 200)
zr = [log(log(signalStrength(xi, yi, uReal, vReal) + 1)) for yi in yr, xi in xr]

# Contour plot background
contour(xr, yr, zr'; ratio=1, colorbar=true, title="Radar Estimation (Improved)", xlabel="x", ylabel="y")

# Initial guess
scatter!(x0[1:4], x0[5:8], label="Initial Guess", color=:red, markersize=6)

# Newton final result
scatter!(ures_newton, vres_newton, label="Newton Result", 
         color=:blue, markersize=8, marker=:diamond)

# Plot Newton steps with transparency
for (idx, step) in enumerate(res_newton.koraki)
    u_step = step[1:4]
    v_step = step[5:8]
    scatter!(u_step, v_step, ms=3, 
             color=RGBA(0.5, 0, 0.8, 0.4), 
             label=idx==1 ? "Newton Steps" : nothing)
end

# True radar positions
scatter!(uReal, vReal, marker=:star5, ms=5, 
         label="True Radar Positions", color=:green)

# Adjust legend properties here
plot!(legend=:topleft, fontsize=5)  # Now we use plot! instead of legend!

xlims!(-7, 7)
ylims!(-7, 7)

savefig("radar_newton_improved.png")
