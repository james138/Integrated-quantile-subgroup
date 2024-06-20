function [gamma,beta1_hat,alpha1_hat,beta2_hat,alpha2_hat,beta3_hat,alpha3_hat,time,no_class,class_id,BIC_loss,BIC_C,BIC]=...
    BIC_for_3Quantile_SCAD(gamma_list,Y1,X1,Z1,Y2,X2,Z2,Y3,X3,Z3,a,lambda_initial,tau,K)
[n,p1] = size(X1);[~,p2] = size(X2);[~,p3] = size(X3);
[~,q1] = size(Z1);[~,q2] = size(Z2);[~,q3] = size(Z3);
sum_p=p1+p2+p3;
sum_q=q1+q2+q3;
pi_1 = 1/3;pi_2 = 1/3;pi_3 = 1/3;
len=size(gamma_list,2);
BIC=zeros(len,1);
BIC_loss=zeros(len,1);
BIC_C=zeros(len,1);
time=zeros(len,1);
beta1=zeros(n,p1,len);beta2=zeros(n,p2,len);beta3=zeros(n,p3,len);
alpha1=zeros(q1,1,len);alpha2=zeros(q2,1,len);alpha3=zeros(q3,1,len);
K_gamma=zeros(len,1);
id=zeros(len,n);


parfor i=1:len

        [beta1(:,:,i),beta2(:,:,i),beta3(:,:,i),alpha1(:,:,i),alpha2(:,:,i),alpha3(:,:,i),D,B,time_per_iter] = ...
            ADMM_3Quantile_SCAD_knn(Y1,X1,Z1,Y2,X2,Z2,Y3,X3,Z3,a,gamma_list(i),lambda_initial,tau,K);


    time(i)=sum(time_per_iter);
    Y1_hat=sum(X1.*beta1(:,:,i),2)+Z1*alpha1(:,:,i);
    Y2_hat=sum(X2.*beta2(:,:,i),2)+Z2*alpha2(:,:,i);
    Y3_hat=sum(X3.*beta3(:,:,i),2)+Z3*alpha3(:,:,i);
    V=round(D*[beta1(:,:,i),beta2(:,:,i),beta3(:,:,i)],4);
    [K_gamma(i),id(i,:)] = group_assign_vertice( V',B,n );
    BIC(i)=n*(pi_1*log(sum(checkloss(Y1-Y1_hat,tau)/n+eps))+pi_2*log(sum(checkloss(Y2-Y2_hat,tau)/n+eps))+pi_3*log(sum(checkloss(Y3-Y3_hat,tau)/n+eps)))+...
        log(log(n*sum_p+sum_q))*log(n)*(K_gamma(i)*sum_p+sum_q);
    BIC_loss(i)=n*(pi_1*log(sum(checkloss(Y1-Y1_hat,tau)/n+eps))+pi_2*log(sum(checkloss(Y2-Y2_hat,tau)/n+eps))+pi_3*log(sum(checkloss(Y3-Y3_hat,tau)/n+eps)));
    BIC_C(i)=log(log(n*sum_p+sum_q))*log(n)*(K_gamma(i)*sum_p+sum_q);

end
[~,I]=min(BIC);
gamma=gamma_list(I);
beta1_hat=beta1(:,:,I);beta2_hat=beta2(:,:,I);beta3_hat=beta3(:,:,I);
alpha1_hat=alpha1(:,:,I);alpha2_hat=alpha2(:,:,I);alpha3_hat=alpha3(:,:,I);
no_class=K_gamma(I);
class_id=id(I,:);

end