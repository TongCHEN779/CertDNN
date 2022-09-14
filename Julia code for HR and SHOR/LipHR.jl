##############################################################################
##### This is the Julia code of HR algorithm for 1-hidden layer networks #####
##############################################################################

using JuMP
using LinearAlgebra
using MosekTools
using MAT
using MATLAB

# function to create moment matrices
function moment_matrix(var, base)
    n = binomial(2+2*2, 2*2)
    vars = matread("Moment_and_Localization_Matrices.mat")
    M_temp = vars["moment_2_2"]
    M = 0
    for t = 1:n
        M = M .+ M_temp[t]*base[t]
    end
    return M
end

# function to create localization matrices
function localization_matrix(g, var, base)
    n = binomial(2+2*2, 2*2); m = binomial(2+2, 2);
    vars = matread("Moment_and_Localization_Matrices.mat")
    M_t = vars["localization_2_1"]; M_temp = Array{Any}(undef, n);
    for i = 1:n
        M_temp[i] = 0
        for j = 1:m
            M_temp[i] = M_temp[i] .+ g[j] * M_t[j][i]
        end
    end
    M = 0
    for t = 1:n
        M = M .+ M_temp[t]*base[t]
    end
    return M
end

# size of the SDP-NN network (784,500)
p0 = 784; p1 = 500;

# A, b, c: parameters; x0: initial input
vars = matread("SDP-NN.mat")
A = vars["A"]; b = vars["b"]; c = vars["c"]; x0 = vars["x00"];

# range of perturbation: ϵ = 10 for global case; ϵ = 0.1 for local case
ϵ = 10;

# index of variables in the first-degree monomial list [1,x,y,z,t]
idx_x = 2:1+p0; idx_y = 2+p0:1+p0+p1; idx_z = 2+p0+p1:1+p0+2*p1; idx_t = 2+p0+2*p1:1+2*p0+2*p1;

# lower and upper bounds of variable x
lx = x0.-ϵ*ones(p0,1); ux = x0.+ϵ*ones(p0,1);
# lower and upper bounds of variable y
ly = 1*(((A .* (A .> 0)) * lx .+ (A .* (A .< 0)) * ux .+ b) .> 0);
uy = 1*(((A .* (A .> 0)) * ux .+ (A .* (A .< 0)) * lx .+ b) .> 0);
# lower and upper bounds of variable z
lz = (A .* (A .> 0)) * lx .+ (A .* (A .< 0)) * ux .+ b;
uz = (A .* (A .> 0)) * ux .+ (A .* (A .< 0)) * lx .+ b;

# build model
model = Model(with_optimizer(Mosek.Optimizer))

# define first-order moment matrix
@variable(model, P[1:1+2*p0+2*p1, 1:1+2*p0+2*p1], Symmetric)

# define objective function
obj = sum(P[idx_t, idx_y] .* (A' * diagm(c[:,1])))
@objective(model, Max, obj)

# define constraints
@constraints(model, begin
    P in PSDCone() # M_1 (y) >= 0
    P[1,1] == 1 # L_y (1) = 1
    P[idx_z, 1]-A*P[idx_x, 1]-b .== 0 # z - Ax - b = 0
end)
# (z - Ax - b)^2 = 0
for i = 1:p1
    ind_z = 1+p0+p1+i
    @constraint(model, P[ind_z, ind_z] - 2*A[i,:]'*P[idx_x, ind_z] - 2*b[i]*P[ind_z, 1] + sum((A[i,:]*A[i,:]') .* (P[idx_x, idx_x])) + 2*b[i]*A[i,:]'*P[idx_x,1] + b[i]^2 == 0)
end
# constraints w.r.t. [x,t]
for i = 1:p0
    ind_x = 1+i; ind_t = 1+p0+2*p1+i
    var = P[[ind_x, ind_t], 1]
    n = binomial(length(var)+2*2, 2*2)
    meas = @variable(model, [1:n,1:1])
    MomentMatrix = moment_matrix(var, meas)
    LocalizationMatrix1 = localization_matrix([-lx[i]*ux[i], lx[i]+ux[i], 0, -1, 0, 0], var, meas) # (x - lx)(x - ux) <= 0
    LocalizationMatrix2 = localization_matrix([1,0,0,0,0,-1], var, meas) # t^2 <= 1
    @constraints(model, begin
        Symmetric(MomentMatrix) in PSDCone()
        Symmetric(LocalizationMatrix1) in PSDCone()
        Symmetric(LocalizationMatrix2) in PSDCone()
        meas[1] .== P[1,1]
        meas[2] .== P[ind_x, 1]
        meas[3] .== P[ind_t, 1]
        meas[4] .== P[ind_x, ind_x]
        meas[5] .== P[ind_x, ind_t]
        meas[6] .== P[ind_t, ind_t]
    end)
end
# constraints w.r.t. [y,z]
for i = 1:p1
    ind_y = 1+p0+i; ind_z = 1+p0+p1+i
    var = P[[ind_y, 1+p0+p1+i], 1]
    n = binomial(length(var)+2*2, 2*2)
    meas = @variable(model, [1:n,1:1])
    MomentMatrix = moment_matrix(var, meas)
    LocalizationMatrix1 = localization_matrix([0,-1,0,1,0,0], var, meas) # y (y - 1) = 0
    LocalizationMatrix2 = localization_matrix([0,0,-1/2,0,1,0], var, meas) # (y - 1/2) z >= 0
    LocalizationMatrix3 = localization_matrix([-ly[i]*uy[i], ly[i]+uy[i], 0, -1, 0, 0], var, meas) # (y - ly)(y - uy) <= 0
    LocalizationMatrix4 = localization_matrix([-lz[i]*uz[i], 0, lz[i]+uz[i], 0, 0, -1], var, meas) # (z - lz)(z - uz) <= 0
    @constraints(model, begin
        Symmetric(MomentMatrix) in PSDCone()
        Symmetric(LocalizationMatrix1) .== 0
        Symmetric(LocalizationMatrix2) in PSDCone()
        Symmetric(LocalizationMatrix3) in PSDCone()
        Symmetric(LocalizationMatrix4) in PSDCone()
        meas[1] .== P[1,1]
        meas[2] .== P[ind_y, 1]
        meas[3] .== P[ind_z, 1]
        meas[4] .== P[ind_y, ind_y]
        meas[5] .== P[ind_y, ind_z]
        meas[6] .== P[ind_z, ind_z]
    end)
end

# solve
optimize!(model)

# print optimal solution
print(objective_value(model))
