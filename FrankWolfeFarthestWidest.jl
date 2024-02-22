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

"""Projects y onto the closed ball with a given center and radius."""
function BallProjection(y::Vector{T}, radius = one(T),
 center = zeros(T, length(y))) where T <: Real
#
  distance = norm(y - center)
  if distance > radius
    return center + (radius / distance) * (y - center)
  else
    return y
  end
end

"""Evaluates the elastic net."""
function elastic_net(x::T, r = one(T)) where T <: Real
#
  return abs(x) + r * x^2 / 2
end

"""Evaluates the separable elastic net."""
function separable_elastic_net(v::Vector{T}) where T <: Real
#
  return sum(elastic_net, v)
end

"""Evaluates the inverse gradient of the constraint f."""
function inverse_gradient(v::Vector{T}, r = one(T)) where T <: Real
#
  n = length(v)
  inv_grad = zeros(T, n)
  for i = 1:n
    inv_grad[i] = sign(v[i]) * (max(abs(v[i]) - 1, zero(T))) / r
  end
  return inv_grad
end

"""Finds the support point of y on the sublevel set {x: f(x) <= c}."""
function SubLevelSupp(y::Vector{T}, c = one(T),
  tol = 1.0e-12) where T <: Real
#
  f = separable_elastic_net
  (a, b) = (one(T), one(T)) # bracket points for the root
  s = f(y / a)
  if s <= c # check if y is in the sublevel set
    return y
  else # otherwise find bracketing interval
    compose = f ∘ inverse_gradient
    s = compose(y / a)
    while s <= c
      a = a / 2
      s = compose(y / a)
    end
    s = compose(y / b)
    while s >= c
      b = 2 * b
      s = compose(y / b)
    end
    h(u) = compose(y / u) - c
    (t, iter) = bisect(h, a, b, tol)
    return inverse_gradient(y / t)
  end
end

"""Finds the support point for y on the ball of radius r and 
given center."""
function BallSupp(y::Vector{T}, radius = one(T),
 center = zeros(T, length(y))) where T <: Real
#
  return center + (radius / norm(y)) * y
end

"""Finds the support point for y on the box [a, b]."""
function BoxSupp(y::Vector{T}, a = -ones(T, length(y)), 
  b = ones(T, length(y))) where T <: Real
#
  n = length(y)
  x = zeros(T, n)
  for i = 1:n
    if y[i] > zero(T)
      x[i] = b[i]
    elseif y[i] < zero(T)
      x[i] = a[i]
    else
      x[i] = (a[i] + b[i]) / 2
    end
  end
  return x
end

"""Finds the support point for y on the simplex {x | x >= 0, sum(x) = r}."""
function SimplexSupp(y::Vector{T}, r = one(T)) where T <: Real
#
  x = zeros(T, length(y))
  (v, m) = findmax(y)
  x[m] = r
  return x
end

"""Finds the support point for y on the L1 ball."""
function L1BallSupp(y::Vector{T}, r = one(T)) where T <: Real
#
  x = zeros(T, length(y))
  (v, m) = findmax(abs, y)
  x[m] = sign(y[m]) * r
  return x
end

"""Finds the support point for y on the intersection of the ball of
radius r and the nonnegative orthant."""
function BallAndOrthantSupp(y::Vector{T}, r = one(T)) where T <: Real
#
  x = max.(y, zero(T))
  if sum(x) <= zero(T)
    return zeros(T, length(y))
  else
    return (r / norm(x)) * x
  end
end

"""Evaluates the support operator of a Cartesian product at y."""
function SeparableSupp(Supp::Function, y::Vector{T}) where T <: Real
#
  n = length(y)
  x = zeros(T, n)
  for i = 1:n
    x[i] = Supp(y[i])
  end
  return x
end

"""Evaluates the support operator of the elastic net."""
function SuppElasticNet(y::T, rho = one(T)) where T <: Real
#
  if abs(y) <= one(T)
    return zero(T)
  else  
    return sign(y) * (abs(y) - one(T)) / rho
  end
end

"""Finds the point supporting z on the Minkowski rounded set R =
c * S + (1 - c) * B. Here B is the unit ball and Supp_S is the support
map onto S."""
function MinkowskiSupp(Supp::Function, z, t)
  return t * Supp(z) + (1 - t) * (1 / norm(z)) * z
end

"""Finds the farthest point on a convex set from a point p."""
function farthest(Supp::Function, x, p)
  tol = 1.0e-8
  for iter = 1:100
    xnew = Supp(x - p)
    conv = norm(xnew - x)
    x .= xnew
    if conv < tol 
      break
    end
  end
  return (norm(x - p), x)
end

"""Finds the diameter of a convex set."""
function widest(Supp::Function, x, y)
  (iters, tol) = (0, 2.0e-8)
  for iter = 1:100
    iters = iters + 1 
    (xnew, ynew) = (Supp(x - y), Supp(y - x))
    conv = norm(xnew - x) + norm(ynew - y)
    x .= xnew
    y .= ynew
    if conv < tol
      break
    end 
  end
  return (norm(x - y), x, y, iters)
end

"""Finds the diameter by homotopy."""
function widest_homotopy(Supp::Function, n)
  x = randn(n)
  x = x /norm(x) # random point on unit sphere
  y = -x  # point on opposite side of unit sphere
  (homotopy_points, tol) = (10, 2.0e-8)
  for iter = 0:homotopy_points
    t = iter / homotopy_points
    for i = 1:100
      xnew = MinkowskiSupp(Supp, x - y, t)
      ynew = MinkowskiSupp(Supp, y - x, t)
      conv = norm(x - xnew) + norm(y - ynew)
      x .= xnew
      y .= ynew
      if conv < tol
        break
      end
    end
  end
  return (norm(x - y), x, y)
end

"""Orchestrates farthest or widest estimation."""
function master(Supp, problem, homotopy, n, trials, io)
  (count, optimum, obj) = (0, 0.0, 0.0)
  if problem == "farthest" p = randn(n) end
  for trial = 1:trials
    if problem == "farthest" 
      x0 = -p
      (obj, x) = farthest(Supp, x0, p)
    elseif problem == "widest"
      if homotopy == "yes"
        (obj, x, y) = widest_homotopy(Supp, n)
      else
        x0 = randn(n)
        x0 = x0 / norm(x0)
        y0 = -x0
        (obj, x, y) = widest(Supp, x0, y0)
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

outfile = "FrankWolfeFarthest.out";
io = open(outfile, "w");
trials = 100;
println(io,"Set"," & ","Dimension"," & ","Type"," & ","Homotopy"," & ","Fraction"," & ",
  "Maximum"," & ","Seconds"," \\ ")
for n in [2, 3, 10, 1000]
  for i = 2:6
    if i == 1
      Supp = BallSupp
      title = "L2 ball"
    elseif i == 2
      Supp = BoxSupp
      title = "box"
    elseif i == 3
      Supp = BallAndOrthantSupp
      title = "ball and orthant"
    elseif i == 4
      Supp = SimplexSupp
      title = "simplex"
    elseif i == 5
      Supp = L1BallSupp
      title = "L1 ball"
    elseif i == 6
      Supp = SubLevelSupp
      title = "elastic net"
    end
    println(title)
    (problem, homotopy) = ("farthest", "no");
    Random.seed!(123)
    time = @elapsed (fraction, optimum) = master(Supp, problem, homotopy, n, trials, io)
    println(io,title," & ",n," & ",problem," & ",homotopy," & ","   ", # round(fraction, sigdigits=3),
      " & ",round(optimum, sigdigits=5)," & ",round(time,sigdigits=3)," \\ ") 
    (problem, homotopy) = ("widest", "no")
    Random.seed!(123)
    time = @elapsed (fraction, optimum) = master(Supp, problem, homotopy, n, trials, io)
    println(io,title," & ",n," & ",problem," & ",homotopy," & ",round(fraction, sigdigits=3),
      " & ",round(optimum, sigdigits=5)," & ",round(time,sigdigits=3)," \\ ")
    homotopy = "yes"
    Random.seed!(123)
    time = @elapsed (fraction, optimum) = master(Supp, problem, homotopy, n, trials, io)
    println(io,title," & ",n," & ",problem," & ",homotopy," & ",round(fraction, sigdigits=3),
      " & ",round(optimum, sigdigits=5)," & ",round(time,sigdigits=3)," \\ ")
  end
end
close(io)
