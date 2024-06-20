function [gamma,beta_hat,alpha_hat,time,no_class,class_id,BIC_loss,BIC_C,BIC]=...
    BIC_for_Quantile_SCAD(gamma_list,Y,X,Z,a,lambda_initial,tau,K)
[n,p] = size(X);
[~,q] = size(Z);
sum_p=p;
sum_q=q;

pi_1 = 1;
len=size(gamma_list,2);
BIC=zeros(len,1);
BIC_loss=zeros(len,1);
BIC_C=zeros(len,1);
time=zeros(len,1);
beta=zeros(n,p,len);
alpha=zeros(q,1,len);
K_gamma=zeros(len,1);
id=zeros(len,n);

len_l=n*(n-1)/2;
Dwhole=zeros(len_l,n);
B=zeros(len_l,2);
k=0;
for i=1:(n-1)
    for j=(i+1):n
        k=k+1;
        Dwhole(k,i)=1;
        Dwhole(k,j)=-1;
        B(k,:)=[i,j];
    end
end
parfor i=1:len
        [beta(:,:,i),alpha(:,:,i),D,B,time_per_iter] = ADMM_Quantile_SCAD_knn(Y,X,Z,a,gamma_list(i),lambda_initial,tau,K);


    time(i)=sum(time_per_iter);
    Y_hat=sum(X.*beta(:,:,i),2)+Z*alpha(:,:,i);
    V=round(D*beta(:,:,i),4);
    [K_gamma(i),id(i,:)] = group_assign_vertice( V',B,n );
    BIC(i)=n*(pi_1*log(sum(checkloss(Y-Y_hat,tau)/n+eps)))+log(log(n*sum_p+sum_q))*log(n)*(K_gamma(i)*sum_p+sum_q);
    BIC_loss(i)=n*(pi_1*log(sum(checkloss(Y-Y_hat,tau)/n+eps)));
    BIC_C(i)=log(log(n*sum_p+sum_q))*log(n)*(K_gamma(i)*sum_p+sum_q);
end
[~,I]=min(BIC);
gamma=gamma_list(I);
beta_hat=beta(:,:,I);
alpha_hat=alpha(:,:,I);
no_class=K_gamma(I);
class_id=id(I,:);

end