using LinearAlgebra
using Plots

# Neznani položaji radarjev (u_k, v_k)
u = [2.0, 4.0, -5.0, 3.0]
v = [-5.0, -5.0, -5.0, -5.0]
# Položaji opravljenih meritev (x_i, y_i)
x = [3.0, 0, -3.0, 0, 0, 0, 3.0, 1.5]
y = [0, -3.0, 0, 3.0, 2.4, -2.4, -2.5]
# Znani položaji radarjev
uReal = [4.0, 0, -4.0, 0]
vReal = [0, -4.0, 0, 4.0]

# Fukcija za izračun vrednosti z_i, jakosti signalov, v točkah (x_i, y_i) na podlagi znanih
# lokacij (uReal_k, vReal_k)
function strengthes(x, y, uReal, vReal)
    res = []
    for (xi, yi) in zip(x, y)
        push!(res, signalStrength(xi, yi, uReal, vReal))
    end
    return res
end

# Fukcija V_i(u, v). Izračuna vsoto jakosti signalov radarjev v točki (x_i, y_i);
function signalStrength(xi, yi, u, v)
    return sum(1 / ((xi - ui)^2 + (yi - vi)^2) for (ui, vi) in zip(u, v))
end

# Funkcija, ki jo minimiziramo v smislu metode najmanjših kvadratov
function F(z, u, v, x, y) 
    return sum((signalStrength(xi, yi, u, v) - zi)^2 for (xi, yi, zi) in zip(x, y, z))
end

# Parcialni odvod j(x_i, y_i) po u_k
function DSignalStrengthU(xi, yi, ui, vi)
    return 2 * (xi - ui) / ((xi - ui)^2 + (yi - vi)^2)^2
end

# Parcialni odvod j(x_i, y_i) po v_k
function DSignalStrengthV(xi, yi, ui, vi)
    return 2 * (yi - vi) / ((xi - ui)^2 + (yi - vi)^2)^2
end

# Gradient funkcije F
function GradF(z, u, v, x, y)
    du = Float64[]
    dv = Float64[]
    for k in 1:length(u)
        du_k = 0.0
        dv_k = 0.0
        for (xi, yi, zi) in zip(x, y, z)
            du_k += 2 * (signalStrength(xi, yi, u, v) - zi) * DSignalStrengthU(xi, yi, u[k], v[k])
            dv_k += 2 * (signalStrength(xi, yi, u, v) - zi) * DSignalStrengthV(xi, yi, u[k], v[k])
        end
        push!(du, du_k)
        push!(dv, dv_k)
    end
    return vcat(du, dv)
end

# Gradientna metoda
function gradmet(z, x, y, alpha, x0; tol = 1e-12, maxit = 10000, record_steps = false)
    n = 1
    x1 = x0
    steps = []

    if record_steps
        u0 = x1[1:length(x0) ÷ 2]
        v0 = x1[length(x0) ÷ 2 + 1:end]
        for (ui, vi) in zip(u0, v0)
            push!(steps, [ui; vi])
        end
    end

    for outer n in 1:maxit
        u0 = x0[1:length(x0) ÷ 2]
        v0 = x0[length(x0) ÷ 2 + 1:end]
        grad = GradF(z, u0, v0, x, y)
        x1 = x0 .- alpha .* grad

        if record_steps
            korakiu = x1[1:length(x0) ÷ 2]
            korakiv = x1[length(x0) ÷ 2 + 1:end]
            for (ui, vi) in zip(korakiu, korakiv)
                push!(steps, [ui; vi])
            end
        end

        if norm(x1 - x0) < alpha * tol
            break
        end

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

# Izračun jakosti z_i v točkah (x_i, y_i).
z = strengthes(x, y, uReal, vReal)
#z .+= 0.01 .* randn(length(z))  # dodamo naključni šum

# Začetni približek
x0 = vcat(u, v)
X, _, koraki = gradmet(z, x, y, 0.01, x0; tol = 1e-10, record_steps = true, maxit = 10000)

ures = X[1:length(u)]
vres = X[length(u)+1:end]

# Narišemo konture za vsoto jakosti signalov
xr = LinRange(-7, 7, 200)
yr = LinRange(-7, 7, 200)
zr = [log(log(signalStrength(xi, yi, uReal, vReal) + 1)) for xi in xr, yi in yr]
contour(xr, yr, zr'; ratio = 1, colorbar=true)

# Dodamo začetne ocene pozicije radarjev
scatter!(u, v, label="Začetni radarji", color=:red)

# Dodamo rezultat izračuna pozicij radarjev
scatter!(ures, vres, label="Ocene radarjev", color=:blue)

# Dodamo korake gradientne metode
scatter!([T[1] for T in koraki], [T[2] for T in koraki], ms=1, label="Koraki", color=:green)