function [Beta,alpha,D,B,time_per_iter] = ADMM_Quantile_SCAD_knn(Y,X,Z,a,lambda,lambda_initial,tau,K)

[n,p] = size(X);
C=num2cell(X,2); %按行
X=blkdiag(C{:});
X=sparse(X);
a1=1;pi_1=1;
a2=1;
len_l=n*(n-1)/2;
Dwhole=zeros(len_l,n);
k=0;
for i=1:(n-1)
    for j=(i+1):n
        k=k+1;
        Dwhole(k,i)=1;
        Dwhole(k,j)=-1;
    end
end
Dwhole=sparse(Dwhole);
% A2=(kron(Dwhole,eye(p2)))';
Q=eye(n)-Z*((Z'*Z)\Z');
XTQ=X'*Q;
Mz = (Z'*Z)\Z';
%beta&alpha    
beta=zeros(n*p,1);
beta_old=ones(n*p,1);

u = zeros(n,1);
delta = zeros(n,1);
M_inv = inv(a1*XTQ*X + lambda_initial*kron(Dwhole'*Dwhole,eye(p)));

%%% Fusion

while norm(beta_old-beta,2)>1e-6

    % theta2 update
    beta_old=beta;
    beta = M_inv * (a1*XTQ*(Y-u+delta));
    alpha=Mz*(Y-u+delta-X*beta);
    % t update
    u = prox(Y-X*beta-Z*alpha+delta,tau,a1);
    
    % s upate
    delta = delta + Y-X*beta-Z*alpha - u;
    
end
Beta=reshape(beta,[p,n])';
[D,~,~,B]=knn(Beta',K,Dwhole');
[len_l,~]=size(D);
D=sparse(D);

%beta&alpha    
beta=zeros(n*p,1);
beta_old=ones(n*p,1);

u = zeros(n,1);
delta = zeros(n,1);
M_inv = inv(a1*XTQ*X + lambda_initial*kron(D'*D,eye(p)));

%%% Fusion

while norm(beta_old-beta,2)>1e-6

    % theta2 update
    beta_old=beta;
    beta = M_inv * (a1*XTQ*(Y-u+delta));
    alpha=Mz*(Y-u+delta-X*beta);
    % t update
    u = prox(Y-X*beta-Z*alpha+delta,tau,a1);
    
    % s upate
    delta = delta + Y-X*beta-Z*alpha - u;
    
end
Beta=reshape(beta,[p,n])';


delta=zeros(n,1);
M_inv = inv(a1*XTQ*X + a2*kron(D'*D,eye(p)) );
u = prox(Y-X*beta-Z*alpha,tau,a1/pi_1);
V = D*Beta;
Gamma = zeros(len_l,p);
V_tilde=V-Gamma;
ATV=V_tilde'*D;
ATV=ATV(:);
% Stroage of Output
MAX_ITER  = 5000;
time_per_iter = zeros(1,MAX_ITER);


for m = 1:MAX_ITER
    
    tic;
    beta_old=beta;
    alpha_old=alpha;
    % theta update
    beta = M_inv*(a1*XTQ*(Y-u+delta) +a2*ATV ) ;
    alpha=Mz*((Y-u+delta-X*beta));
    % u update
    u = prox(Y-X*beta-Z*alpha+delta,tau,a1/pi_1);
    % V update
    Beta=reshape(beta,[p,n])';
    DB=D*Beta;
    V=DB+Gamma;
    V=mexFunction(V,lambda, a);
    % delta upate
    delta = delta + Y-X*beta-Z*alpha - u;
    % Gamma update
    Gamma=Gamma-V+DB;
    V_tilde=V-Gamma;
    ATV=V_tilde'*D;
    ATV=ATV(:);
    time_per_iter(m) = toc;
    
    if m > 10 && (norm(beta_old-beta,'fro')+norm(alpha_old-alpha,'fro') < 1e-6)

        break
    end
    
    
end

time_per_iter = time_per_iter(1:m);

end

 

