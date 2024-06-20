function[beta,alpha]=oracle_est_balance(Y,X,Z,K,tau)
[n,p] = size(X);[~,q] = size(Z);

t = zeros(n,1);
s = zeros(n,1);
Xo=[];
for i=1:K
    Xo=blkdiag(Xo,X(((i-1)*n/K+1):(i*n/K),:));
end

X_tmp=[Xo,Z];
xz_old=zeros(K*p+q,1);
xz=ones(K*p+q,1);
while norm(xz_old-xz,2)>1e-6
    xz_old=xz;
    xz=(X_tmp'*X_tmp)\X_tmp'*(Y-t+s);
    t = prox(Y+s-X_tmp*xz,tau,n*1);
    s = s + Y-X_tmp*xz - t;
end
beta=reshape(xz(1:K*p),[p,K])';
alpha=xz((K*p+1):(K*p+q));

end