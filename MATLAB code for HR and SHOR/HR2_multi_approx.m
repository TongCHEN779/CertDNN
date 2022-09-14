function obj_time = HR2_multi_approx(A1, A2, b1, b2, c, x00, type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% This is the MATLAB code of HR (approx) algorithm for two-hidden layer networks %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
p0 = size(A1,2); p1 = size(A1,1); p2 = size(A2,1); % t,x0: d_in; x1,y1,z1: d1; y2,z2: d2;
%% index of variables in the first degree monomial list [1,t,x0,x1,y1,y2,z1,z2]
idx_t = 2:1+p0; idx_x0 = 2+p0:1+2*p0; idx_x1 = 2+2*p0:1+2*p0+p1; 
idx_y1 = 2+2*p0+p1:1+2*p0+2*p1; idx_y2 = 2+2*p0+2*p1:1+2*p0+2*p1+p2;
idx_z1 = 2+2*p0+2*p1+p2:1+2*p0+3*p1+p2; idx_z2 = 2+2*p0+3*p1+p2:1+2*p0+3*p1+2*p2;
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
P = sdpvar(1+2*p0+3*p1+2*p2, 1+2*p0+3*p1+2*p2, 'symmetric');
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
    diag(P(idx_y2, idx_y2))-(ly2+uy2).*P(idx_y2, 1)+ly2.*uy2<=0, ... % (y2-ly2)(y2-uy2)<=0
    diag(P(idx_z1,idx_z1)) - ...
        2*diag(A1*P(idx_x0,idx_z1)) - ...
        2*b1.*P(idx_z1,1) + ...
        diag(A1*P(idx_x0,idx_x0)*A1') + ...
        2*b1.*(A1*P(idx_x0,1)) + b1.*b1==0, ... % (z1-A1x0-b1)^2 = 0
    P(idx_z1,1)-A1*P(idx_x0,1)-b1==0, ... % z1-A1x0-b1 = 0
    diag(P(idx_z2,idx_z2)) - ...
        2*diag(A2*P(idx_x1,idx_z2)) - ...
        2*b2.*P(idx_z2,1) + ...
        diag(A2*P(idx_x1,idx_x1)*A2') + ...
        2*b2.*(A2*P(idx_x1,1)) + b2.*b2==0, ... % (z2-A2x1-b2)^2 = 0
    P(idx_z2,1)-A2*P(idx_x1,1)-b2==0]; % z2-A2x1-b2 = 0
% constraints w.r.t. [x0, t]
for i = 1:p0
    ind_x0 = 1+p0+i; ind_t = 1+i;
    var = P([ind_x0,ind_t], 1);
    n = nchoosek(length(var)+2*2, 2*2);
    meas = [P(1,1);P(ind_x0,1);P(ind_t,1);P(ind_x0,ind_x0);P(ind_x0,ind_t);P(ind_t,ind_t);sdpvar(n-6,1)];
    MomentMatrix = moment_matrix(var, 2, meas);
    LocalizationMatrix1 = localization_matrix([-lx0(i)*ux0(i),lx0(i)+ux0(i),0,-1,0,0], var, 1, meas); % (x0-lx0)(x0-ux0)<=0
    LocalizationMatrix2 = localization_matrix([0,0,1,0,0,-1], var, 1, meas); % t(t-1)<=0
    constraints = [constraints, MomentMatrix>=0, ...
        LocalizationMatrix1>=0, ...
        LocalizationMatrix2>=0];
end
% constraints w.r.t. [x1, z1] & [y1, z1]
for i = 1:p1
    ind_x1 = 1+2*p0+i; ind_y1 = 1+2*p0+p1+i; ind_z1 = 1+2*p0+2*p1+p2+i;
    var1 = P([ind_x1,ind_z1], 1); var2 = P([ind_y1,ind_z1], 1);
    n1 = nchoosek(length(var1)+2*2, 2*2);
    meas1 = [P(1,1);P(ind_x1,1);P(ind_z1,1);P(ind_x1,ind_x1);P(ind_x1,ind_z1);P(ind_z1,ind_z1);sdpvar(n1-6,1)];
    meas2 = [P(1,1);P(ind_y1,1);P(ind_z1,1);P(ind_y1,ind_y1);P(ind_y1,ind_z1);P(ind_z1,ind_z1);sdpvar(3,1);meas1(10);sdpvar(4,1);meas1(15)];
    MomentMatrix1 = moment_matrix(var1, 2, meas1);
    MomentMatrix2 = moment_matrix(var2, 2, meas2);
    LocalizationMatrix11 = localization_matrix([0,0,0,1,-1,0], var1, 1, meas1); % x1*(x1-z1)==0
    LocalizationMatrix12 = localization_matrix([0,1,-1,0,0,0], var1, 1, meas1); % x1-z1>=0
    LocalizationMatrix13 = localization_matrix([0,1,0,0,0,0], var1, 1, meas1); % x1>=0
    LocalizationMatrix14 = localization_matrix([-lx1(i)*ux1(i),lx1(i)+ux1(i),0,-1,0,0], var1, 1, meas1); % (x1-lx1)(x1-ux1)<=0
    LocalizationMatrix21 = localization_matrix([0,-1,0,1,0,0], var2, 1, meas2); % y1*(y1-1)==0
    LocalizationMatrix22 = localization_matrix([0,0,-1/2,0,1,0], var2, 1, meas2); % (y1-1/2)*z1>=0
    LocalizationMatrix23 = localization_matrix([-ly1(i)*uy1(i),ly1(i)+uy1(i),0,-1,0,0], var2, 1, meas2); % (y1-ly1)(y1-uy1)<=0
    LocalizationMatrix24 = localization_matrix([-lz1(i)*uz1(i),0,lz1(i)+uz1(i),0,0,-1], var2, 1, meas2); % (z1-lz1)(z-uz1)<=0
    constraints = [constraints, MomentMatrix1>=0, MomentMatrix2>=0, ...
        LocalizationMatrix11==0, ...
        LocalizationMatrix12>=0, ...
        LocalizationMatrix13>=0, ...
        LocalizationMatrix14>=0, ...
        LocalizationMatrix21==0, ...
        LocalizationMatrix22>=0, ...
        LocalizationMatrix23>=0, ...
        LocalizationMatrix24>=0];
end
% constraints w.r.t. [y2, z2]
for i = 1:p2
    ind_y2 = 1+2*p0+2*p1+i; ind_z2 = 1+2*p0+3*p1+p2+i;
    var = P([ind_y2,ind_z2], 1);
    n = nchoosek(length(var)+2*2, 2*2);
    meas = [P(1,1);P(ind_y2,1);P(ind_z2,1);P(ind_y2,ind_y2);P(ind_y2,ind_z2);P(ind_z2,ind_z2);sdpvar(n-6,1)];
    MomentMatrix = moment_matrix(var, 2, meas);
    LocalizationMatrix1 = localization_matrix([0,-1,0,1,0,0], var, 1, meas); % y2*(y2-1)==0
    LocalizationMatrix2 = localization_matrix([0,0,-1/2,0,1,0], var, 1, meas); % (y2-1/2)*z2>=0
    LocalizationMatrix3 = localization_matrix([-ly2(i)*uy2(i),ly2(i)+uy2(i),0,-1,0,0], var, 1, meas); % (y2-ly2)(y2-uy2)<=0
    LocalizationMatrix4 = localization_matrix([-lz2(i)*uz2(i),0,lz2(i)+uz2(i),0,0,-1], var, 1, meas); % (z2-lz2)(z2-uz2)<=0
    constraints = [constraints, MomentMatrix>=0, ...
        LocalizationMatrix1==0, ...
        LocalizationMatrix2>=0, ...
        LocalizationMatrix3>=0, ...
        LocalizationMatrix4>=0];
end
%% solve
options = sdpsettings('solver', 'mosek', 'dualize', 1);
sol = optimize(constraints, objective, options);
%% optimal solution and solving time
obj_time.obj = value(-objective);
obj_time.time = sol.solvertime;
end