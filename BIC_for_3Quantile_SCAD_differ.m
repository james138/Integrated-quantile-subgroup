function [lambda1,lambda2,beta1_hat,beta2_hat,beta3_hat,time,no_class,class_id,homo]=...
    BIC_for_3Quantile_SCAD_differ(lambda1_list,lambda2_list,Y1,X1,Y2,X2,Y3,X3,a,lambda_initial,tau,K)
[n,p1] = size(X1);[~,p2] = size(X2);[~,p3] = size(X3);
sum_p=p1+p2+p3;
pi_1 = 1/3;pi_2 = 1/3;pi_3 = 1/3;
len1=size(lambda1_list,2);len2=size(lambda2_list,2);
BIC1=zeros(len1,1);BIC2=zeros(len2,1);
BIC_loss=zeros(len1,1);
BIC_C=zeros(len1,1);
time=zeros(len1+len2,1);
beta1=zeros(n,p1,len1);beta2=zeros(n,p2,len1);beta3=zeros(n,p3,len1);
homo=zeros(len2,3);
K_lambda1=zeros(len1,1);K_lambda2=zeros(len2,1);
id=zeros(len1,n);
E=eye(n)-ones(n,1)*ones(1,n)/n;

parfor i=1:len1

    [beta1(:,:,i),beta2(:,:,i),beta3(:,:,i),D,B,time_per_iter] = ...
        ADMM_3Quantile_SCAD_knn_difer(Y1,X1,Y2,X2,Y3,X3,a,lambda1_list(i),0.5,lambda_initial,tau,K);
    time(i)=sum(time_per_iter);
    Y1_hat=sum(X1.*beta1(:,:,i),2);
    Y2_hat=sum(X2.*beta2(:,:,i),2);
    Y3_hat=sum(X3.*beta3(:,:,i),2);
    V=round(D*[beta1(:,:,i),beta2(:,:,i),beta3(:,:,i)],4);
    [K_lambda1(i),id(i,:)] = group_assign_vertice( V',B,n );
    kpq1=KPplusQinBIC(E,beta1(:,:,i),p1,K_lambda1(i));
    kpq2=KPplusQinBIC(E,beta2(:,:,i),p2,K_lambda1(i));
    kpq3=KPplusQinBIC(E,beta3(:,:,i),p3,K_lambda1(i));
    
    BIC1(i)=n*(pi_1*log(sum(checkloss(Y1-Y1_hat,tau)/n+eps))+pi_2*log(sum(checkloss(Y2-Y2_hat,tau)/n+eps))+pi_3*log(sum(checkloss(Y3-Y3_hat,tau)/n+eps)))+...
        log(log(n*sum_p))*log(n)*(kpq1+kpq2+kpq3);
    BIC_loss(i)=n*(pi_1*log(sum(checkloss(Y1-Y1_hat,tau)/n+eps))+pi_2*log(sum(checkloss(Y2-Y2_hat,tau)/n+eps))+pi_3*log(sum(checkloss(Y3-Y3_hat,tau)/n+eps)));
    BIC_C(i)=log(log(n*sum_p))*log(n)*(kpq1+kpq2+kpq3);
    %
end
[~,I]=min(BIC1);
lambda1=lambda1_list(I);
parfor i=1:len2

    [beta1(:,:,i),beta2(:,:,i),beta3(:,:,i),D,B,time_per_iter] = ...
        ADMM_3Quantile_SCAD_knn_difer(Y1,X1,Y2,X2,Y3,X3,a,lambda1,lambda2_list(i),lambda_initial,tau,K);
    time(len1+i)=sum(time_per_iter);
    Y1_hat=sum(X1.*beta1(:,:,i),2);
    Y2_hat=sum(X2.*beta2(:,:,i),2);
    Y3_hat=sum(X3.*beta3(:,:,i),2);
    V=round(D*[beta1(:,:,i),beta2(:,:,i),beta3(:,:,i)],4);
    [K_lambda2(i),id(i,:)] = group_assign_vertice( V',B,n );

    kpq1=KPplusQinBIC(E,beta1(:,:,i),p1,K_lambda2(i));
    kpq2=KPplusQinBIC(E,beta2(:,:,i),p2,K_lambda2(i));
    kpq3=KPplusQinBIC(E,beta3(:,:,i),p3,K_lambda2(i));
    
    homo(i,:)=[kpq1==p1,kpq2==p2,kpq3==p3];
    BIC2(i)=n*(pi_1*log(sum(checkloss(Y1-Y1_hat,tau)/n+eps))+pi_2*log(sum(checkloss(Y2-Y2_hat,tau)/n+eps))+pi_3*log(sum(checkloss(Y3-Y3_hat,tau)/n+eps)))+...
        log(log(n*sum_p))*log(n)*(kpq1+kpq2+kpq3);
    BIC_loss(i)=n*(pi_1*log(sum(checkloss(Y1-Y1_hat,tau)/n+eps))+pi_2*log(sum(checkloss(Y2-Y2_hat,tau)/n+eps))+pi_3*log(sum(checkloss(Y3-Y3_hat,tau)/n+eps)));
    BIC_C(i)=log(log(n*sum_p))*log(n)*(kpq1+kpq2+kpq3);
    %
end
[~,I]=min(BIC2);
lambda2=lambda2_list(I);
beta1_hat=beta1(:,:,I);beta2_hat=beta2(:,:,I);beta3_hat=beta3(:,:,I);
no_class=K_lambda2(I);
class_id=id(I,:);
homo=homo(I,:);
end