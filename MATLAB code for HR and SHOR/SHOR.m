function obj_time = SHOR(A, b, c, x00, type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% This is the MATLAB code of SHOR algorithm for one-hidden layer networks %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
p0 = size(A,2); p1 = size(A,1); % x,t: p0; y: p1
%% index of variables in the first degree monomial list [1,x,y,t]
idx_x = 2:1+p0; idx_y = 2+p0:1+p0+p1; idx_t = 2+p0+p1:1+2*p0+p1;
%% lower and upper bounds of variables
% x
lx = x00 - eps; ux = x00 + eps;
% z
lz = max(A, 0)*lx + min(A, 0)*ux + b;
uz = max(A, 0)*ux + min(A, 0)*lx + b;
% y
ly = 1 * (lz > 0); uy = 1 * (uz > 0);
%% first-order moment matrix
P = sdpvar(1+2*p0+p1, 1+2*p0+p1, 'symmetric');
%% define objective
objective = -sum(sum((A'*diag(c)).*P(idx_t, idx_y)));
%% define constraints
constraints = [P>=0, ... % M_1 (y) >= 0
    P(1,1)==1, ... % L_y (1) = 1
    diag(P(idx_y, idx_y))==P(idx_y, 1), ... % y*(y-1)==0
    2*diag(A*P(idx_x, idx_y))+2*b.*P(idx_y, 1)>=A*P(idx_x, 1)+b, ... % (y-1/2)*(Ax+b)>=0
    diag(P(idx_t, idx_t))<=1, ... %P(idx_z,1)>=0, P(idx_z,1)<=1, ... % 0<=t<=1, t^2<=1
    diag(P(idx_y, idx_y))-(ly+uy).*P(idx_y, 1)+ly.*uy<=0, ...% (y-ly)(y-uy)<=0
    diag(P(idx_x, idx_x))-(lx+ux).*P(idx_x, 1)+lx.*ux<=0]; % (x-lx)(x-ux)<=0
%% solve
options = sdpsettings('solver', 'mosek', 'dualize', 1);
sol = optimize(constraints, objective, options);
%% optimal solution and solving time
obj_time.obj = value(-objective);
obj_time.time = sol.solvertime;
end