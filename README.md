# Integrated-quantile-subgroup
## Example 1
It is recommended to use 'parpool' to speed up the execution on multi-core machines.
```
%setting 
n=200;tau=0.5;f=5;
p=2;q=2;
p1=p;p2=p;p3=p;q1=q;q2=q;q3=q;
beta11=[1;1];beta12=[-1;-1];alpha11=[1;1];
beta21=[1;1];beta22=[-1;-1];alpha22=[1;1];
beta31=[1;1];beta32=[-1;-1];alpha33=[1;1];
% parameter
lambda1_list=[0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.5,2,4];
lambda2_list=[0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.5,2,4];
a=3.7;
lambda_initial=0.0001;
k=2;K=10;
% true label
true_id=[ones(1,n/2),ones(1,n/2)*2];
E=eye(n)-ones(n,1)*ones(1,n)/n;
% output
iter=200; 
RMSE_ora=zeros(iter,6);

record_gamma_1q=zeros(iter,1);
record_gamma_3q_differ=zeros(iter,2);
record_time_1q=zeros(iter,1);
record_time_3q_differ=zeros(iter,1);
record_RMSE_1q=zeros(iter,2);
record_RMSE_3q_differ=zeros(iter,6);
record_noclass_1q=zeros(iter,1);
record_noclass_3q_differ=zeros(iter,1);
record_ARI_RI_1q=zeros(iter,2);
record_ARI_RI_3q_differ=zeros(iter,2);
record_Prop_FN_FP_3q_differ=zeros(iter,3);

record_gamma_3q=zeros(iter,1);
record_time_3q=zeros(iter,1);
record_RMSE_3q=zeros(iter,6);
record_noclass_3q=zeros(iter,1);
record_ARI_RI_3q=zeros(iter,2);

fprintf('n= %d \n',n);


for i=1:iter
    fprintf('i= %d \n',i);
%   --------------------------------------------------------------------------------------------------------------
%   -------------------------------------------%Quantile--------------------------------------------------------
%   --------------------------------------------------------------------------------------------------------------
    [Y1,Y2,Y3,X1,X2,X3,Z1,Z2,Z3]=...
    data_generator(n,p1,p2,p3,q1,q2,q3,beta11,beta12,beta21,beta22,beta31,beta32,alpha11,alpha22,alpha33,tau,f,i);
    [beta1,alpha1]=oracle_est_balance(Y1,X1,Z1,k,tau);
    [beta2,alpha2]=oracle_est_balance(Y2,X2,Z2,k,tau);
    [beta3,alpha3]=oracle_est_balance(Y3,X3,Z3,k,tau);
    b11=beta1(1,:)';b12=beta1(2,:)';b21=beta2(1,:)';b22=beta2(2,:)';b31=beta3(1,:)';b32=beta3(2,:)';
    mseo_b1=sqrt((norm(b11-beta11,'fro')^2+norm(b12-beta12,'fro')^2)/(2*p1));
    mseo_b2=sqrt((norm(b21-beta21,'fro')^2+norm(b22-beta22,'fro')^2)/(2*p2));
    mseo_b3=sqrt((norm(b31-beta31,'fro')^2+norm(b32-beta32,'fro')^2)/(2*p3));
    mseo_a1=sqrt(norm(alpha1-alpha11,2)^2/(q1));
    mseo_a2=sqrt(norm(alpha2-alpha22,2)^2/(q2));
    mseo_a3=sqrt(norm(alpha3-alpha33,2)^2/(q3));
    RMSE_ora(i,:)=[mseo_a1,mseo_a2,mseo_a3,mseo_b1,mseo_b2,mseo_b3];

    % 3q differ
    [record_gamma_3q_differ(i,1),record_gamma_3q_differ(i,2),beta1,beta2,beta3,time_per_iter,record_noclass_3q_differ(i),class_id]=...
    BIC_for_3Quantile_SCAD_differ(lambda1_list,lambda2_list,Y1,[X1,Z1],Y2,[X2,Z2],Y3,[X3,Z3],a,lambda_initial,tau,K);
    record_time_3q_differ(i,1)=sum(time_per_iter);
    mse_b1=sqrt((norm(beta1(1:n/2,1:p1)-beta11','fro')^2+norm(beta1((n/2+1):n,1:p1)-beta12','fro')^2)/(n*p1));
    mse_b2=sqrt((norm(beta2(1:n/2,1:p2)-beta21','fro')^2+norm(beta2((n/2+1):n,1:p2)-beta22','fro')^2)/(n*p2));
    mse_b3=sqrt((norm(beta3(1:n/2,1:p3)-beta31','fro')^2+norm(beta3((n/2+1):n,1:p3)-beta32','fro')^2)/(n*p3));
    mse_a1=sqrt(norm(beta1(:,(p1+1):(p1+q1))-alpha11',2)^2/(n*q1));
    mse_a2=sqrt(norm(beta2(:,(p2+1):(p2+q2))-alpha22',2)^2/(n*q2));
    mse_a3=sqrt(norm(beta3(:,(p3+1):(p3+q3))-alpha33',2)^2/(n*q3));
    record_RMSE_3q_differ(i,:)=[mse_a1,mse_a2,mse_a3,mse_b1,mse_b2,mse_b3];
    [ARI,RI]=RandIndex(true_id,class_id);
    record_ARI_RI_3q_differ(i,:)=[ARI,RI];
    FN1_cnt=p1-sum(all(vecnorm(round(E*beta1(:,1:p1),4),2,1),1));FP1_cnt=sum(all(vecnorm(round(E*beta1(:,(p1+1):(p1+q1)),4),2,1),1));
    FN2_cnt=p2-sum(all(vecnorm(round(E*beta2(:,1:p2),4),2,1),1));FP2_cnt=sum(all(vecnorm(round(E*beta2(:,(p2+1):(p2+q2)),4),2,1),1));
    FN3_cnt=p3-sum(all(vecnorm(round(E*beta3(:,1:p3),4),2,1),1));FP3_cnt=sum(all(vecnorm(round(E*beta3(:,(p3+1):(p3+q3)),4),2,1),1));
    FN_cnt=FN1_cnt+FN2_cnt+FN3_cnt;FP_cnt=FP1_cnt+FP2_cnt+FP3_cnt;
    record_Prop_FN_FP_3q_differ(i,1)=(FN_cnt==0)&&(FP_cnt==0);
    record_Prop_FN_FP_3q_differ(i,2)=FN_cnt/(p1+p2+p3);
    record_Prop_FN_FP_3q_differ(i,3)=FP_cnt/(q1+q2+q3);
    
    % 1q
    [record_gamma_1q(i,1),beta1,alpha1,time_per_iter,record_noclass_1q(i),class_id]=...
        BIC_for_Quantile_SCAD(lambda1_list,Y1,X1,Z1,a,lambda_initial,tau,K);
    mse_b1=sqrt((norm(beta1(1:n/2,:)-beta11','fro')^2+norm(beta1((n/2+1):n,:)-beta12','fro')^2)/(n*p1));
    mse_a1=sqrt(norm(alpha1-alpha11,2)^2/(q1));   
    record_time_1q(i,1)=sum(time_per_iter);
    record_RMSE_1q(i,:)=[mse_a1,mse_b1];
    [ARI,RI]=RandIndex(true_id,class_id);
    record_ARI_RI_1q(i,:)=[ARI,RI];

    % 3q
    [record_gamma_3q(i),beta1,alpha1,beta2,alpha2,beta3,alpha3,time_per_iter,record_noclass_3q(i),class_id]=...
    BIC_for_3Quantile_SCAD(lambda1_list,Y1,X1,Z1,Y2,X2,Z2,Y3,X3,Z3,a,lambda_initial,tau,K);
    record_time_3q(i,1)=sum(time_per_iter);
    mse_b1=sqrt((norm(beta1(1:n/2,:)-beta11','fro')^2+norm(beta1((n/2+1):n,:)-beta12','fro')^2)/(n*p1));
    mse_b2=sqrt((norm(beta2(1:n/2,:)-beta21','fro')^2+norm(beta2((n/2+1):n,:)-beta22','fro')^2)/(n*p2));
    mse_b3=sqrt((norm(beta3(1:n/2,:)-beta31','fro')^2+norm(beta3((n/2+1):n,:)-beta32','fro')^2)/(n*p3));
    mse_a1=sqrt(norm(alpha1-alpha11,2)^2/(q1));
    mse_a2=sqrt(norm(alpha2-alpha22,2)^2/(q2));
    mse_a3=sqrt(norm(alpha3-alpha33,2)^2/(q3));
    record_RMSE_3q(i,:)=[mse_a1,mse_a2,mse_a3,mse_b1,mse_b2,mse_b3];
    [ARI,RI]=RandIndex(true_id,class_id);
    record_ARI_RI_3q(i,:)=[ARI,RI];

end

r=3;row=8;
col_alpha1=strings(row,1);col_alpha2=strings(row,1);col_alpha3=strings(row,1);
col_beta1=strings(row,1);col_beta2=strings(row,1);col_beta3=strings(row,1);
col_RI=strings(row,1);col_ARI=strings(row,1);col_K=strings(row,1);col_perK=strings(row,1);

col_alpha1(1)=round(mean(RMSE_ora(:,1)),r);
col_alpha1(2)='('+string(round(std(RMSE_ora(:,1)),r))+')';
col_alpha1(3)=round(mean(record_RMSE_1q(:,1)),r);
col_alpha1(4)='('+string(round(std(record_RMSE_1q(:,1)),r))+')';
col_alpha1(7)=round(mean(record_RMSE_3q(:,1)),r);
col_alpha1(8)='('+string(round(std(record_RMSE_3q(:,1)),r))+')';
col_alpha1(5)=round(mean(record_RMSE_3q_differ(:,1)),r);
col_alpha1(6)='('+string(round(std(record_RMSE_3q_differ(:,1)),r))+')';


col_alpha2(1)=round(mean(RMSE_ora(:,2)),r);
col_alpha2(2)='('+string(round(std(RMSE_ora(:,2)),r))+')';
col_alpha2(3)='-';
col_alpha2(4)='-';
col_alpha2(7)=round(mean(record_RMSE_3q(:,2)),r);
col_alpha2(8)='('+string(round(std(record_RMSE_3q(:,2)),r))+')';
col_alpha2(5)=round(mean(record_RMSE_3q_differ(:,2)),r);
col_alpha2(6)='('+string(round(std(record_RMSE_3q_differ(:,2)),r))+')';


col_alpha3(1)=round(mean(RMSE_ora(:,3)),r);
col_alpha3(2)='('+string(round(std(RMSE_ora(:,3)),r))+')';
col_alpha3(3)='-';
col_alpha3(4)='-';
col_alpha3(7)=round(mean(record_RMSE_3q(:,3)),r);
col_alpha3(8)='('+string(round(std(record_RMSE_3q(:,3)),r))+')';
col_alpha3(5)=round(mean(record_RMSE_3q_differ(:,3)),r);
col_alpha3(6)='('+string(round(std(record_RMSE_3q_differ(:,3)),r))+')';


col_beta1(1)=round(mean(RMSE_ora(:,4)),r);
col_beta1(2)='('+string(round(std(RMSE_ora(:,4)),r))+')';
col_beta1(3)=round(mean(record_RMSE_1q(:,2)),r);
col_beta1(4)='('+string(round(std(record_RMSE_1q(:,2)),r))+')';
col_beta1(7)=round(mean(record_RMSE_3q(:,4)),r);
col_beta1(8)='('+string(round(std(record_RMSE_3q(:,4)),r))+')';
col_beta1(5)=round(mean(record_RMSE_3q_differ(:,4)),r);
col_beta1(6)='('+string(round(std(record_RMSE_3q_differ(:,4)),r))+')';


col_beta2(1)=round(mean(RMSE_ora(:,5)),r);
col_beta2(2)='('+string(round(std(RMSE_ora(:,5)),r))+')';
col_beta2(3)='-';
col_beta2(4)='-';
col_beta2(7)=round(mean(record_RMSE_3q(:,5)),r);
col_beta2(8)='('+string(round(std(record_RMSE_3q(:,5)),r))+')';
col_beta2(5)=round(mean(record_RMSE_3q_differ(:,5)),r);
col_beta2(6)='('+string(round(std(record_RMSE_3q_differ(:,5)),r))+')';


col_beta3(1)=round(mean(RMSE_ora(:,6)),r);
col_beta3(2)='('+string(round(std(RMSE_ora(:,6)),r))+')';
col_beta3(3)='-';
col_beta3(4)='-';
col_beta3(7)=round(mean(record_RMSE_3q(:,6)),r);
col_beta3(8)='('+string(round(std(record_RMSE_3q(:,6)),r))+')';
col_beta3(5)=round(mean(record_RMSE_3q_differ(:,6)),r);
col_beta3(6)='('+string(round(std(record_RMSE_3q_differ(:,6)),r))+')';


col_RI(1)='-';
col_RI(2)='-';
col_RI(3)=round(mean(record_ARI_RI_1q(:,2)),r);
col_RI(4)='('+string(round(std(record_ARI_RI_1q(:,2)),r))+')';
col_RI(7)=round(mean(record_ARI_RI_3q(:,2)),r);
col_RI(8)='('+string(round(std(record_ARI_RI_3q(:,2)),r))+')';
col_RI(5)=round(mean(record_ARI_RI_3q_differ(:,2)),r);
col_RI(6)='('+string(round(std(record_ARI_RI_3q_differ(:,2)),r))+')';


col_ARI(1)='-';
col_ARI(2)='-';
col_ARI(3)=round(mean(record_ARI_RI_1q(:,1)),r);
col_ARI(4)='('+string(round(std(record_ARI_RI_1q(:,1)),r))+')';
col_ARI(7)=round(mean(record_ARI_RI_3q(:,1)),r);
col_ARI(8)='('+string(round(std(record_ARI_RI_3q(:,1)),r))+')';
col_ARI(5)=round(mean(record_ARI_RI_3q_differ(:,1)),r);
col_ARI(6)='('+string(round(std(record_ARI_RI_3q_differ(:,1)),r))+')';


col_K(1)='-';
col_K(2)='-';
col_K(3)=round(mean(record_noclass_1q(:,1)),r);
col_K(4)='('+string(round(std(record_noclass_1q(:,1)),r))+')';
col_K(7)=round(mean(record_noclass_3q(:,1)),r);
col_K(8)='('+string(round(std(record_noclass_3q(:,1)),r))+')';
col_K(5)=round(mean(record_noclass_3q_differ(:,1)),r);
col_K(6)='('+string(round(std(record_noclass_3q_differ(:,1)),r))+')';


col_perK(1)='-';
col_perK(2)=' ';
col_perK(3)=string(round(sum(record_noclass_1q(:,1)==k)/iter*100,1))+'\%';
col_perK(4)=' ';
col_perK(7)=string(round(sum(record_noclass_3q(:,1)==k)/iter*100,1))+'\%';
col_perK(8)=' ';
col_perK(5)=string(round(sum(record_noclass_3q_differ(:,1)==k)/iter*100,1))+'\%';
col_perK(6)=' ';

T = table(col_alpha1,col_alpha2,col_alpha3,col_beta1,col_beta2,col_beta3,col_RI,col_ARI,col_K,col_perK);
```
