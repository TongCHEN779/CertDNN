function obj_time = SHOR_multi_approx(A1, A2, b1, b2, c, x00, type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% This is the MATLAB code of SHOR (approx) algorithm for two-hidden layer networks %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% A1, A2: weight matrices
%% b1, b2: biases
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
%% size of networks (p0, p1, p2)
p0 = size(A1,2); p1 = size(A1,1); p2 = size(A2,1); % t, x0: p0; x1, y1: p1; y2: p2;
%% index of variables in the first degree monomial list [1,t,x0,x1,y1,y2]
idx_t = 2:1+p0; idx_x0 = 2+p0:1+2*p0; idx_x1 = 2+2*p0:1+2*p0+p1; idx_y1 = 2+2*p0+p1:1+2*p0+2*p1; idx_y2 = 2+2*p0+2*p1:1+2*p0+2*p1+p2;
%% lower and upper bounds of variables
% x0
lx0 = x00 - eps; ux0 = x00 + eps;
% z1
lz1 = max(A1, 0)*lx0 + min(A1, 0)*ux0 + b1;
uz1 = max(A1, 0)*ux0 + min(A1, 0)*lx0 + b1;
% y1
ly1 = 1 * (lz1 > 0); uy1 = 1 * (uz1 > 0);
% x1
lx1 = max(lz1, 0); ux1 = max(uz1, 0);
% z2
lz2 = max(A2, 0)*lx1 + min(A2, 0)*ux1 + b2;
uz2 = max(A2, 0)*ux1 + min(A2, 0)*lx1 + b2;
% y2
ly2 = 1 * (lz2 > 0); uy2 = 1 * (uz2 > 0);
%% first-order moment matrix
P = sdpvar(1+2*p0+2*p1+p2, 1+2*p0+2*p1+p2, 'symmetric');
%% define objective by approximation technique
sum1 = sum(sum((A1' * diag(A2' * c)) .* P(idx_t, idx_y1))); % t*y1'
sum2 = sum(sum((diag(ones(1,p0) * A1') * A2' * diag(c)) .* P(idx_y1, idx_y2))); % y1*y2'
sum3 = sum(sum((A1' * A2' * diag(c)) .* P(idx_t, idx_y2))); % t*y2'
sum4 = P(idx_t, 1)' * A1' * A2' * c; % t
sum5 = sum(sum(A1' * diag((A2' * c) .* P(idx_y1, 1)))); % y1
sum6 = sum(sum(A1' * diag(A2' * (c .* P(idx_y2, 1))))); % y2
sum7 = sum(sum(0.25 * (A1' * diag(A2' * c)) .* (A1' * diag(A2' * c) >= 0)));
sum0 = 0;
for i = 1:p0
    sum0 = sum0 + sum(sum((diag(A1(:,i)) * A2' * diag(c)) .* P(idx_y1, idx_y2)));
end
objective = - (sum1 + sum2 + sum3) + 0.5 * (sum4 + sum5 + sum6) - 2 * sum7 + sum0;
%% define constraints
constraints = [P>=0, ... % M_1 (y) >= 0
    P(1,1)==1, ... % L_y (1) = 1
    diag(P(idx_y1, idx_y1))==P(idx_y1, 1), ... % y1*(y1-1)==0
    diag(P(idx_y2, idx_y2))==P(idx_y2, 1), ... % y2*(y2-1)==0
    2*diag(A1*P(idx_x0, idx_y1))+2*b1.*P(idx_y1, 1)>=A1*P(idx_x0, 1)+b1, ... % (y1-1/2)*(A1x0+b1)>=0
    2*diag(A2*P(idx_x1, idx_y2))+2*b2.*P(idx_y2, 1)>=A2*P(idx_x1, 1)+b2, ... % (y2-1/2)*(A2x1+b2)>=0
    diag(P(idx_x1, idx_x1))-diag(A1*P(idx_x0, idx_x1))-b1.*P(idx_x1, 1)==0, ... % x1(x1-A1x0-b1)=0
    P(idx_x1, 1)>=A1*P(idx_x0, 1)+b1, ... % x1>=A1x0+b1
    P(idx_x1, 1)>=0, ... % x1>=0
    diag(P(idx_t, idx_t))<=1, ... % t^2<=1
    diag(P(idx_x0, idx_x0))-(lx0+ux0).*P(idx_x0, 1)+lx0.*ux0<=0, ... % (x0-lx0)(x0-ux0)<=0
    diag(P(idx_y1, idx_y1))-(ly1+uy1).*P(idx_y1, 1)+ly1.*uy1<=0, ... % (y1-ly1)(y1-uy1)<=0
    diag(P(idx_x1, idx_x1))-(lx1+ux1).*P(idx_x1, 1)+lx1.*ux1<=0, ... % (x1-lx1)(x1-ux1)<=0
    diag(P(idx_y2, idx_y2))-(ly2+uy2).*P(idx_y2, 1)+ly2.*uy2<=0]; % (y2-ly2)(y2-uy2)<=0
%% solve
options = sdpsettings('solver', 'mosek', 'dualize', 1);
sol = optimize(constraints, objective, options);
%% optimal solution and solving time
obj_time.obj = value(-objective);
obj_time.time = sol.solvertime;
end