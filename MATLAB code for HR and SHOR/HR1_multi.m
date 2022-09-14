function obj_time = HR1_multi(A1, A2, b1, b2, c, x00, type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% This is the MATLAB code of SHOR algorithm for two-hidden layer networks %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
ly2 = 1 * (lz2 > 0); uy2 = 1 * (uz2 > 0);%% first-order moment matrix
P = sdpvar(1+2*p0+2*p1+p2, 1+2*p0+2*p1+p2, 'symmetric');
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
%% define objective adding sub-moment matrices
objective = 0;
for k = 1:p2
    ind_y2 = 1+2*p0+2*p1+k;
    MomentBlock1 = sdpvar(p0, p1, 'full');
    MomentBlock2 = sdpvar(p1, p1, 'symmetric');
    MomentMatrix = [1, P(1,idx_t), P(ind_y2, idx_y1); ...
        P(idx_t, 1), P(idx_t, idx_t), MomentBlock1; ...
        P(idx_y1, ind_y2), MomentBlock1', MomentBlock2];
    for i = 1:p0
        for j = 1:p1
            l = min([-ly1(j)*ly2(k), -ly1(j)*uy2(k), -uy1(j)*ly2(k), -uy1(j)*uy2(k), ...
                ly1(j)*ly2(k), ly1(j)*uy2(k), uy1(j)*ly2(k), uy1(j)*uy2(k)]);
            u = max([-ly1(j)*ly2(k), -ly1(j)*uy2(k), -uy1(j)*ly2(k), -uy1(j)*uy2(k), ...
                ly1(j)*ly2(k), ly1(j)*uy2(k), uy1(j)*ly2(k), uy1(j)*uy2(k)]);
            objective = objective - c(k)*A2(k,j)*A1(j,i)*MomentBlock1(i,j);
            constraints = [constraints, ...
                MomentBlock1(i,j)<=u, ...
                MomentBlock1(i,j)>=l];
        end
    end 
    for i = 1:p1
        for j = 1:p1
            l = min([ly1(i)*ly1(j)*ly2(k)^2, ly1(i)*ly1(j)*uy2(k)^2, ly1(i)*uy1(j)*ly2(k)^2, ly1(i)*uy1(j)*uy2(k)^2, ...
                uy1(i)*ly1(j)*ly2(k)^2, uy1(i)*ly1(j)*uy2(k)^2, uy1(i)*uy1(j)*ly2(k)^2, uy1(i)*uy1(j)*uy2(k)^2]);
            u = max([ly1(i)*ly1(j)*ly2(k)^2, ly1(i)*ly1(j)*uy2(k)^2, ly1(i)*uy1(j)*ly2(k)^2, ly1(i)*uy1(j)*uy2(k)^2, ...
                uy1(i)*ly1(j)*ly2(k)^2, uy1(i)*ly1(j)*uy2(k)^2, uy1(i)*uy1(j)*ly2(k)^2, uy1(i)*uy1(j)*uy2(k)^2]);
            constraints = [constraints, ...
                MomentBlock2(i,j)<=u, ...
                MomentBlock2(i,j)>=l];
        end
    end
    constraints = [constraints, MomentMatrix>=0];
end
%% solve
options = sdpsettings('solver', 'mosek', 'dualize', 1);
sol = optimize(constraints, objective, options);
%% optimal solution and solving time
obj_time.obj = value(-objective);
obj_time.time = sol.solvertime;
end