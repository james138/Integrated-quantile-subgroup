function[Y1,Y2,Y3,X1,X2,X3,Z1,Z2,Z3]=...
    data_generator(n,p1,p2,p3,q1,q2,q3,beta11,beta12,beta21,beta22,beta31,beta32,alpha11,alpha22,alpha33,tau,df,i)
rng('default');
rng(i);

e0=icdf('T',tau,df);
XZ1=[ones([n,1]),mvnrnd(zeros(p1+q1-1,1), AR(p1+q1-1, 0.5), n)];
XZ2=[ones([n,1]),mvnrnd(zeros(p2+q2-1,1), AR(p2+q2-1, 0.5), n)];
XZ3=[ones([n,1]),mvnrnd(zeros(p3+q3-1,1), AR(p3+q3-1, 0.5), n)];
X1=XZ1(:,1:p1);X2=XZ2(:,1:p2);X3=XZ3(:,1:p3);
Z1=XZ1(:,(1+p1):(p1+q1));Z2=XZ2(:,(1+p2):(p2+q2));Z3=XZ3(:,(1+p3):(p3+q3));
e=mvtrnd(diag(ones(3,1))*0.5+ones(3)*0.5, df,n);
e1=(e(:,1)-e0)*0.3;
e2=(e(:,2)-e0)*0.3;
e3=(e(:,3)-e0)*0.3;


C1=num2cell(X1,2);C2=num2cell(X2,2);C3=num2cell(X3,2);
X1_block=blkdiag(C1{:});X2_block=blkdiag(C2{:});X3_block=blkdiag(C3{:});

Beta1=[repmat(beta11',[1,n/2]),repmat(beta12',[1,n/2])]';
Beta2=[repmat(beta21',[1,n/2]),repmat(beta22',[1,n/2])]';
Beta3=[repmat(beta31',[1,n/2]),repmat(beta32',[1,n/2])]';

Y1=X1_block*Beta1+Z1*alpha11+e1;
Y2=X2_block*Beta2+Z2*alpha22+e2;
Y3=X3_block*Beta3+Z3*alpha33+e3;
end

