function obj_time = HR2(A, b, c, x00, type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% This is the MATLAB code of HR algorithm for one-hidden layer networks %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% A: weight matrices
%% b: biases
%% c: score weight
%% x00: initial input
%% epsilon: range of perturbation
%% type: 0 = global; 1 = local
%%
%% global/local Lipschitz bound
if type == 1
    eps = 0.1;
elseif type == 0
    eps = 10;
end
%% size of networks (p0, p1)
p0 = size(A,2); p1 = size(A,1); % x,t: d_in; y,z: d
%% index of variables in the first degree monomial list [1,x,y,z,t]
idx_x = 2:1+p0; idx_y = 2+p0:1+p0+p1; idx_z = 2+p0+p1:1+p0+2*p1; idx_t = 2+p0+2*p1:1+2*p0+2*p1;
%% lower and upper bounds of variables
% x
lx = x00 - eps; ux = x00 + eps;
% z
lz = max(A, 0)*lx + min(A, 0)*ux + b;
uz = max(A, 0)*ux + min(A, 0)*lx + b;
% y
ly = 1 * (lz > 0); uy = 1 * (uz > 0);
%% first-order moment matrix
P = sdpvar(1+2*p0+2*p1, 1+2*p0+2*p1, 'symmetric');
%% define objective
objective = -sum(sum((A'*diag(c)).*P(idx_t,idx_y)));
%% define constraints
constraints = [P>=0, ... % M_1 (y) >= 0
    P(1,1)==1, ... % L_y (1) = 1
    diag(P(idx_y, idx_y))==P(idx_y, 1), ... % y*(y-1)==0
    2*diag(A*P(idx_x, idx_y))+2*b.*P(idx_y, 1)>=A*P(idx_x, 1)+b, ... % (y-1/2)*(Ax+b)>=0
    diag(P(idx_t, idx_t))<=1, ... %P(idx_z,1)>=0, P(idx_z,1)<=1, ... % 0<=t<=1, t^2<=1
    diag(P(idx_y, idx_y))-(ly+uy).*P(idx_y, 1)+ly.*uy<=0, ...% (y-ly)(y-uy)<=0
    diag(P(idx_x, idx_x))-(lx+ux).*P(idx_x, 1)+lx.*ux<=0, ... % (x-lx)(x-ux)<=0
    diag(P(idx_z,idx_z)) - ...
        2*diag(A*P(idx_x,idx_z)) - ...
        2*b.*P(idx_z,1) + ...
        diag(A*P(idx_x,idx_x)*A') + ...
        2*b.*(A*P(idx_x,1)) + b.*b==0, ... % (z-Ax-b)^2=0
    P(idx_z,1)-A*P(idx_x,1)-b==0]; % z-Ax-b=0
% constraints w.r.t. [x, t]
for i = 1:p0
    var = P([1+i,1+p0+2*p1+i], 1);
    meas = sdpvar(nchoosek(length(var)+2*2, 2*2), 1);
    MomentMatrix = moment_matrix(var, 2, meas);
    LocalizationMatrix1 = localization_matrix([-lx(i)*ux(i),lx(i)+ux(i),0,-1,0,0], var, 1, meas); % (x-lx)(x-ux)<=0
    LocalizationMatrix2 = localization_matrix([1,0,0,0,0,-1], var, 1, meas); % t^2<=1
    constraints = [constraints, MomentMatrix>=0, LocalizationMatrix1>=0, LocalizationMatrix2>=0, ...
        meas(1)==P(1,1),...
        meas(2)==P(1+i,1),...
        meas(3)==P(1+p0+2*p1+i,1),...
        meas(4)==P(1+i,1+i),...
        meas(5)==P(1+i,1+p0+2*p1+i),...
        meas(6)==P(1+p0+2*p1+i,1+p0+2*p1+i)];
end
% constraints w.r.t. [y, z]
for i = 1:p1
    var = P([1+p0+i,1+p0+p1+i], 1);
    meas = sdpvar(nchoosek(length(var)+2*2, 2*2), 1);
    MomentMatrix = moment_matrix(var, 2, meas);
    LocalizationMatrix1 = localization_matrix([0,-1,0,1,0,0], var, 1, meas); % y*(y-1)==0
    LocalizationMatrix2 = localization_matrix([0,0,-1/2,0,1,0], var, 1, meas); % (y-1/2)*z>=0
    LocalizationMatrix3 = localization_matrix([-ly(i)*uy(i),ly(i)+uy(i),0,-1,0,0], var, 1, meas); % (y-ly)(y-uy)<=0
    LocalizationMatrix4 = localization_matrix([-lz(i)*uz(i),0,lz(i)+uz(i),0,0,-1], var, 1, meas); % (z-lz)(z-uz)<=0
    constraints = [constraints, MomentMatrix>=0, LocalizationMatrix1==0, LocalizationMatrix2>=0,  LocalizationMatrix3>=0, LocalizationMatrix4>=0, ...
        meas(1)==P(1,1),...
        meas(2)==P(1+p0+i,1),...
        meas(3)==P(1+p0+p1+i,1),...
        meas(4)==P(1+p0+i,1+p0+i),...
        meas(5)==P(1+p0+i,1+p0+p1+i),...
        meas(6)==P(1+p0+p1+i,1+p0+p1+i)];
end
%% solve
options = sdpsettings('solver', 'mosek', 'dualize', 1);
sol = optimize(constraints, objective, options);
%% optimal solution and solving time
obj_time.obj = value(-objective);
obj_time.time = sol.solvertime;
end