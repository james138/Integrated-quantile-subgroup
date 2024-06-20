function [Beta1,Beta2,Beta3,alpha1,alpha2,alpha3,D,B,time_per_iter] = ADMM_3Quantile_SCAD_knn(Y1,X1,Z1,Y2,X2,Z2,Y3,X3,Z3,a,lambda,lambda_initial,tau,K)

[n,p1] = size(X1);
C1=num2cell(X1,2); %按行
X1=blkdiag(C1{:});
X1=sparse(X1);
[~,p2] = size(X2);
C2=num2cell(X2,2); %按行
X2=blkdiag(C2{:});
X2=sparse(X2);
[~,p3] = size(X3);
C3=num2cell(X3,2); %按行
X3=blkdiag(C3{:});
X3=sparse(X3);
pi_1=1/3;pi_2=1/3;pi_3=1/3;
a1=1;a2=1;
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
Q1=eye(n)-Z1*((Z1'*Z1)\Z1');
Q2=eye(n)-Z2*((Z2'*Z2)\Z2');
Q3=eye(n)-Z3*((Z3'*Z3)\Z3');
XTQ1=X1'*Q1;
XTQ2=X2'*Q2;
XTQ3=X3'*Q3;
Mz1 = (Z1'*Z1)\Z1';
Mz2 = (Z2'*Z2)\Z2';
Mz3 = (Z3'*Z3)\Z3';
%beta&alpha    
beta1=zeros(n*p1,1);beta2=zeros(n*p2,1);beta3=zeros(n*p3,1);
beta1_old=ones(n*p1,1);beta2_old=ones(n*p2,1);beta3_old=ones(n*p3,1);
u1 = zeros(n,1);u2 = zeros(n,1);u3 = zeros(n,1);
delta1 = zeros(n,1);delta2 = zeros(n,1);delta3 = zeros(n,1);
M_inv1 = inv(a1*XTQ1*X1 + lambda_initial*kron(Dwhole'*Dwhole,eye(p1)));
M_inv2 = inv(a1*XTQ2*X2 + lambda_initial*kron(Dwhole'*Dwhole,eye(p2)));
M_inv3 = inv(a1*XTQ3*X3 + lambda_initial*kron(Dwhole'*Dwhole,eye(p3)));

while norm(beta1_old-beta1,2)>1e-6 || norm(beta2_old-beta2,2)>1e-6 || norm(beta3_old-beta3,2)>1e-6

    % beta alpha update
    beta1_old=beta1;beta2_old=beta2;beta3_old=beta3;
    beta1 = M_inv1 * (a1*XTQ1*(Y1-u1+delta1));beta2 = M_inv2 * (a1*XTQ2*(Y2-u2+delta2));beta3 = M_inv3 * (a1*XTQ3*(Y3-u3+delta3));
    alpha1=Mz1*(Y1-u1+delta1-X1*beta1);alpha2=Mz2*(Y2-u2+delta2-X2*beta2);alpha3=Mz3*(Y3-u3+delta3-X3*beta3);
    % u update
    u1 = prox(Y1-X1*beta1-Z1*alpha1+delta1,tau,a1);
    u2 = prox(Y2-X2*beta2-Z2*alpha2+delta2,tau,a1);
    u3 = prox(Y3-X3*beta3-Z3*alpha3+delta3,tau,a1);
    % delta upate
    delta1 = delta1 + Y1-X1*beta1-Z1*alpha1 - u1;
    delta2 = delta2 + Y2-X2*beta2-Z2*alpha2 - u2;
    delta3 = delta3 + Y3-X3*beta3-Z3*alpha3 - u3;
    
end
Beta1=reshape(beta1,[p1,n])';
Beta2=reshape(beta2,[p2,n])';
Beta3=reshape(beta3,[p3,n])';
[D,~,~,B]=knn([Beta1';Beta2';Beta3'],K,Dwhole');
[len_l,~]=size(D);
D=sparse(D);
%beta&alpha    
beta1=zeros(n*p1,1);beta2=zeros(n*p2,1);beta3=zeros(n*p3,1);
beta1_old=ones(n*p1,1);beta2_old=ones(n*p2,1);beta3_old=ones(n*p3,1);
u1 = zeros(n,1);u2 = zeros(n,1);u3 = zeros(n,1);
delta1 = zeros(n,1);delta2 = zeros(n,1);delta3 = zeros(n,1);
M_inv1 = inv(a1*XTQ1*X1 + lambda_initial*kron(D'*D,eye(p1)));
M_inv2 = inv(a1*XTQ2*X2 + lambda_initial*kron(D'*D,eye(p2)));
M_inv3 = inv(a1*XTQ3*X3 + lambda_initial*kron(D'*D,eye(p3)));

while norm(beta1_old-beta1,2)>1e-6 || norm(beta2_old-beta2,2)>1e-6 || norm(beta3_old-beta3,2)>1e-6

    % beta alpha update
    beta1_old=beta1;beta2_old=beta2;beta3_old=beta3;
    beta1 = M_inv1 * (a1*XTQ1*(Y1-u1+delta1));beta2 = M_inv2 * (a1*XTQ2*(Y2-u2+delta2));beta3 = M_inv3 * (a1*XTQ3*(Y3-u3+delta3));
    alpha1=Mz1*(Y1-u1+delta1-X1*beta1);alpha2=Mz2*(Y2-u2+delta2-X2*beta2);alpha3=Mz3*(Y3-u3+delta3-X3*beta3);
    % u update
    u1 = prox(Y1-X1*beta1-Z1*alpha1+delta1,tau,a1);
    u2 = prox(Y2-X2*beta2-Z2*alpha2+delta2,tau,a1);
    u3 = prox(Y3-X3*beta3-Z3*alpha3+delta3,tau,a1);
    % delta upate
    delta1 = delta1 + Y1-X1*beta1-Z1*alpha1 - u1;
    delta2 = delta2 + Y2-X2*beta2-Z2*alpha2 - u2;
    delta3 = delta3 + Y3-X3*beta3-Z3*alpha3 - u3;
    
end
Beta1=reshape(beta1,[p1,n])';
Beta2=reshape(beta2,[p2,n])';
Beta3=reshape(beta3,[p3,n])';

delta1=zeros(n,1);delta2=zeros(n,1);delta3=zeros(n,1);
M_inv1 = inv(a1*XTQ1*X1 + a2*kron(D'*D,eye(p1)));
M_inv2 = inv(a1*XTQ2*X2 + a2*kron(D'*D,eye(p2)));
M_inv3 = inv(a1*XTQ3*X3 + a2*kron(D'*D,eye(p3)));
u1 = prox(Y1-X1*beta1-Z1*alpha1,tau,a1/pi_1);
u2 = prox(Y2-X2*beta2-Z2*alpha2,tau,a1/pi_2);
u3 = prox(Y3-X3*beta3-Z3*alpha3,tau,a1/pi_3);  
V1 = D*Beta1;V2 = D*Beta2;V3 = D*Beta3;
Gamma1 = zeros(len_l,p1);Gamma2 = zeros(len_l,p2);Gamma3 = zeros(len_l,p3);
V1_tilde=V1-Gamma1;V2_tilde=V2-Gamma2;V3_tilde=V3-Gamma3;
% ATM=A'*M_2(:);
ATV1=V1_tilde'*D;ATV2=V2_tilde'*D;ATV3=V3_tilde'*D;
ATV1=ATV1(:);ATV2=ATV2(:);ATV3=ATV3(:);
% Stroage of Output
MAX_ITER  = 5000;
time_per_iter = zeros(1,MAX_ITER);

for m = 1:MAX_ITER
    
    tic;
    beta1_old=beta1;beta2_old=beta2;beta3_old=beta3;
    alpha1_old=alpha1;alpha2_old=alpha2;alpha3_old=alpha3;
    % beta alpha update
    beta1 = M_inv1*(a1*XTQ1*(Y1-u1+delta1) +a2*ATV1 );
    beta2 = M_inv2*(a1*XTQ2*(Y2-u2+delta2) +a2*ATV2 );
    beta3 = M_inv3*(a1*XTQ3*(Y3-u3+delta3) +a2*ATV3 );
    alpha1=Mz1*((Y1-u1+delta1-X1*beta1));
    alpha2=Mz2*((Y2-u2+delta2-X2*beta2));
    alpha3=Mz3*((Y3-u3+delta3-X3*beta3));
    % u update
    u1 = prox(Y1-X1*beta1-Z1*alpha1+delta1,tau,a1/pi_1);
    u2 = prox(Y2-X2*beta2-Z2*alpha2+delta2,tau,a1/pi_2);
    u3 = prox(Y3-X3*beta3-Z3*alpha3+delta3,tau,a1/pi_3);  
    % V update
    Beta1=reshape(beta1,[p1,n])';
    Beta2=reshape(beta2,[p2,n])';
    Beta3=reshape(beta3,[p3,n])';
    DB1=D*Beta1;DB2=D*Beta2;DB3=D*Beta3;
    V1=DB1+Gamma1;V2=DB2+Gamma2;V3=DB3+Gamma3;
    V=[V1,V2,V3];
    V=mexFunction(V,lambda, a);
    V1=V(:,1:p1);
    V2=V(:,(1+p1):(p1+p2));
    V3=V(:,(1+p1+p2):(p1+p2+p3));
    % delta upate
    delta1 = delta1 + Y1-X1*beta1-Z1*alpha1 - u1;
    delta2 = delta2 + Y2-X2*beta2-Z2*alpha2 - u2;
    delta3 = delta3 + Y3-X3*beta3-Z3*alpha3 - u3;    
    % Gamma update
    Gamma1=Gamma1-V1+DB1;
    Gamma2=Gamma2-V2+DB2;
    Gamma3=Gamma3-V3+DB3;
    V1_tilde=V1-Gamma1;
    V2_tilde=V2-Gamma2;
    V3_tilde=V3-Gamma3;
    ATV1=V1_tilde'*D;
    ATV2=V2_tilde'*D;
    ATV3=V3_tilde'*D;
    ATV1=ATV1(:);
    ATV2=ATV2(:);
    ATV3=ATV3(:);
    time_per_iter(m) = toc;
    
    if m > 10 && (norm(beta1_old-beta1,'fro')+norm(alpha1_old-alpha1,'fro') < 1e-6) && (norm(beta2_old-beta2,'fro')+norm(alpha2_old-alpha2,'fro') < 1e-6)&& (norm(beta3_old-beta3,'fro')+norm(alpha3_old-alpha3,'fro') < 1e-6)

        break
    end
    
    
end

time_per_iter = time_per_iter(1:m);

end
