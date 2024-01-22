using LinearAlgebra, Random

""" Implements the bisection algorithm for finding a root of
the equation f(x)=0. The constants a < b should bracket a 0."""
function bisect(f::Function, a::T, b::T, tol::T) where T <: Real
  (fa, fb) = (f(a), f(b))
  @assert(a < b) # check for input error
  @assert(fa * fb <= zero(T)) # check for input error
  for iteration = 1:100
    m = (a + b) / 2
    fm = f(m)
    if abs(fm) < tol
      return (m, iteration)
    end
    if fa * fm < zero(T) 
      (b, fb) = (m, fm)
    else
      (a, fa) = (m, fm)
    end
  end
  return ((a + b) / 2, 100)
end
 
"""Projects y onto the sublevel set {x: f(x) <= c}."""
function SubLevelProjection(f::Function, prox_f::Function, y::Vector{T}, 
  c::T, tol::T) where T <: Real
#
  (a, b) = (one(T), one(T)) # bracket points for the root
  s = f(y, a)
  if s <= c # check if y is in the sublevel set
    return y
  else # otherwise find bracketing interval
    compose = f ∘ prox_f
    s = compose(y, a)
    while s < c
      a = a / 2
      s = compose(y, a)
    end
    s = compose(y, b)
    while s > c
      b = 2 * b
      s = compose(y, b)
    end
    h(u) = compose(y, u) - c
    (t, iter) = bisect(h, a, b, tol)
    return prox_f(y, t)
  end
end

function ElasticNetProjection(y::Vector{T}, c = one(T)) where T <: Real
  tol = 1.0e-12
  enp = separable_elastic_net
  prox_enp(v, t) = SeparableProx(prox_elastic_net, v, t) 
  x = SubLevelProjection(enp, prox_enp, y, c, tol)
  return x
end

"""Evaluates the elastic net."""
function elastic_net(x::T, r = one(T)) where T <: Real
#
  return abs(x) + r * x^2 / 2
end

"""Evaluates the proximal map of the elastic net."""
function prox_elastic_net(y::T, t = one(T), r = one(T)) where T <: Real
#
  return sign(y) * max(abs(y) - t, zero(T)) / (1 + t * r)
end

"""Evaluates the separable elastic net."""
function separable_elastic_net(v::Vector{T}, t = one(T)) where T <: Real
#
  return t * sum(elastic_net, v)
end

"""Evaluates the proximal operator of sum_i t * f(x_i) at y."""
function SeparableProx(prox_f::Function, y::Vector{T}, t::T) where T <: Real
#
  n = length(y)
  x = zeros(T, n)
  for i = 1:n
    x[i] = prox_f(y[i], t)
  end
  return x
end

"""Projects y onto the closed ball with a given center and radius."""
function BallProjection(y::Vector{T}, center = zeros(T, length(y)), 
 radius = one(T)) where T <: Real
#
  distance = norm(y - center)
  if distance > radius
    return center + (radius / distance) * (y - center)
  else
    return y
  end
end

"""Projects the point y onto the closed box with bounds a and b."""
function BoxProjection(y::Vector{T}, a = -ones(T, length(y)),
  b = ones(T, length(y))) where T <: Real
#
  return clamp.(y, a, b)
end

"""Projects the point y onto the simplex {x | x >= 0, sum(x) = r}."""
function SimplexProjection(y::Vector{T}, r = one(T)) where T <: Real
#
  n = length(y)
  z = sort(y, rev = true)
  (s, lambda) = (zero(T), zero(T))
  for i = 1:n
    s = s + z[i]
    lambda = (s - r) / i
    if i < n && lambda < z[i] && lambda >= z[i + 1]
      break
    end
  end
  return max.(y .- lambda, zero(T))
end

"""Projects the point y onto the ell_1 ball with the given 
center and radius."""
function L1BallProjection(y::Vector{T}, center = zeros(T, length(y)),
  radius = one(T)) where T <: Real
#
  p = abs.(y - center)
  if norm(p, 1) <= radius
    return y
  else
    x = SimplexProjection(p, radius)
    return center + sign.(y - center) .* x
  end
end

"""Projects the point y onto the intersection of the ball of
radius r and the nonnegative orthant."""
function BallAndOrthantProjection(y::Vector{T}, r = one(T)) where T <: Real
#
  x = copy(y)
  x = max.(x, zero(T)) # project onto orthant
  return x = (r / max(norm(x), r)) .* x # contract as needed
end

"""Projects the point z onto the Minkowski rounded set
c * S + (1 - c) * B. Here B is the unit ball, Proj 
is projection onto S, and Proj(z) = a + b."""
function MinkowskiNear(Proj, a, b, z, c, conv)
  tol = 2.0e-8
  for iter = 1:100
    anew = c .* Proj((z - b) ./ c) 
    bnew = (1 - c) .* BallProjection((z - anew) ./ (1 - c)) 
    conv = norm(a - anew) + norm(b - bnew)
    a .= anew
    b .= bnew
    if conv < tol
      break 
    end
  end
  return (a, b)
end

"""Finds the farthest point on a convex set from a point p."""
function farthest(Proj::Function, x, p)
  tol = 1.0e-8
  for iter = 1:100
    xnew = Proj(2x - p)
    conv = norm(xnew - x)
    x .= xnew
    if conv < tol 
      break
    end
  end
  return (norm(x - p), x)
end

"""Finds the diameter of a convex set."""
function widest(Proj::Function, x, y)
  tol = 2.0e-8
  for iter = 1:100
    (xnew, ynew) = (Proj(3x / 2 - y / 2), Proj(3y / 2 - x))
    conv = norm(xnew - x) + norm(ynew - y)
    x .= xnew
    y .= ynew
    if conv < tol
      break
    end 
  end
  return (norm(x - y), x, y)
end

"""Finds the diameter by homotopy."""
function widest_homotopy(Proj, n)
 (a, c) = (zeros(n), zeros(n))
  b = randn(n) 
  b = b / norm(b) # random point on unit sphere
  d = -b  # point on opposite side of unit sphere
  (x, y) = (copy(b), copy(d))
  (homotopy_points, conv) = (10, 1.0e-10)
  for iter = 0:homotopy_points
    t = clamp(iter / homotopy_points, 0.000001, 0.999999)
    (a, b) = MinkowskiNear(Proj, a, b, 3x / 2 - y / 2, t, conv)
    (c, d) = MinkowskiNear(Proj, c, d, 3y / 2 - x / 2, t, conv)
    x .= a .+ b
    y .= c .+ d
  end
  return (norm(x - y), x, y)
end 
 
"""Orchestrates farthest or widest testing."""
function master(Proj, problem, homotopy, n, trials, io)
  (count, optimum, obj) = (0, 0.0, 0.0)
  if problem == "farthest" p = randn(n) end
  for trial = 1:trials
    if problem == "farthest" 
      x0 = Proj(-p)
      (obj, x) = farthest(Proj, x0, p)
    elseif problem == "widest"
      if homotopy == "yes"
        (obj, x, y) = widest_homotopy(Proj, n)
      else
        x0 = Proj(randn(n))
        x0 = x0 / norm(x0)
        y0 = -x0
        (obj, x, y) = widest(Proj, x0, y0)
      end
    else
      println("Improper problem choice!")
    end
    if obj > optimum + 10.0e-10
      count = 1
      optimum = obj
    elseif obj > optimum - 10.0e-10
      count = count + 1
    end
    println(problem," homotopy =",homotopy," trial = ",trial," dim = ",n) 
  end
  return (fraction = count / trials, optimum)
end

outfile = "ProjGradFarthest.out";
io = open(outfile, "w");
trials = 100;
println(io,"Set"," & ","Dimension"," & ","Type"," & ","Homotopy"," & ","Fraction"," & ",
  "Maximum"," & ","Seconds"," \\ ")
for n in [2, 3, 10, 1000]
  for i = 2:6
    if i == 1
      Proj = BallProjection
      title = "L2 ball"
    elseif i == 2
      Proj = BoxProjection
      title = "box"
    elseif i == 3
      Proj = BallAndOrthantProjection
      title = "ball and orthant"
    elseif i == 4
      Proj = SimplexProjection
      title = "simplex"
    elseif i == 5
      Proj = L1BallProjection
      title = "L1 ball"
    elseif i == 6
      Proj = ElasticNetProjection
      title = "elastic net"
    end
    println(title)
    (problem, homotopy) = ("farthest", "no");
    Random.seed!(123)
    time = @elapsed (fraction, optimum) = master(Proj, problem, homotopy, n, trials, io)
    println(io,title," & ",n," & ",problem," & ",homotopy," & ","   ", # round(fraction, sigdigits=3),
      " & ",round(optimum, sigdigits=5)," & ",round(time,sigdigits=3)," \\ ") 
    (problem, homotopy) = ("widest", "no")
    Random.seed!(123)
    time = @elapsed (fraction, optimum) = master(Proj, problem, homotopy, n, trials, io)
    println(io,title," & ",n," & ",problem," & ",homotopy," & ",round(fraction, sigdigits=3),
      " & ",round(optimum, sigdigits=5)," & ",round(time,sigdigits=3)," \\ ")
    homotopy = "yes"
    Random.seed!(123)
    time = @elapsed (fraction, optimum) = master(Proj, problem, homotopy, n, trials, io)
    println(io,title," & ",n," & ",problem," & ",homotopy," & ",round(fraction, sigdigits=3),
      " & ",round(optimum, sigdigits=5)," & ",round(time,sigdigits=3)," \\ ")
  end
end
close(io)
