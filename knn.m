function [D_knn,w,DD,B] = knn(beta,K,D_whole)
% Pick 1:K nearest neighbors in distance, for same distance, select
% randomly one

[~,n]  = size(beta);
% [p2 n]  = size(Y);

% Total length of w
w=zeros(n*K,2);
D_knn=zeros(n*K,n);
DD=zeros(n,n);
B=zeros(n*K,2);
% KNN method
    
    tmp = vecnorm(beta*D_whole,1,1);     
    
    % Make it a distance matrix
    D = squareform(tmp);
    % Distance matrix with sparse outputs(only contain K nearest neighbor)
    M = zeros(n*K,2); 

    %%% For each node i, pick its K nearest neighbor
    for i = 1:n
        
        % sort_ind: the distance index from smallest to largest
        [~, sort_ind] = sort(D(i,:),'ascend');     
        
        %%% Select even values in distance with the Kth one
        
        even_index = find( D(i,:) == D(i,sort_ind(K+1)) ); % select index which equals to the Kth minimum distance
        
        if length(even_index) > 1  % If there are ties
            strict_less_count = sum( D(i,:) < D(i,sort_ind(K+1)));  % Find those who is strict nearest neighbor
            sort_ind(strict_less_count+1:K+1) = randsample( even_index, K+1 - strict_less_count);  % For those who have ties, randomly select some so we have K nearest neigobur
            % update the K-nearest neighbor index
        end
        %%%            
        
        % choose 2 to K+1 since the first(smallest) is 0 , distance with itself D(i,i) is also included 
        M(((i-1)*K+1):i*K,2) = sort_ind(2:K+1);
        M(((i-1)*K+1):i*K,1)=i;
        DD(i,sort_ind(2:K+1))=1 ;    
    end
    
    % make M symmetric, since for previous one, i is j's K nearest neighbor
    % does not mean j is i's K nearest neighbor

    l=0;
    for i=1:(n-1)
        for j=(i+1):n
            
            if DD(i,j)+DD(j,i)==2
                l=l+1;
                w(l)=1/(D(i,j)+eps);
                D_knn(l,i)=1;
                D_knn(l,j)=-1;
                B(l,1)=i;
                B(l,2)=j;
            elseif DD(i,j)+DD(j,i)==1
                l=l+1;
                w(l)=1/(2*D(i,j)+eps);
                D_knn(l,i)=1;
                D_knn(l,j)=-1;
                B(l,1)=i;
                B(l,2)=j;
            end
        end
    end
    
    w = w(1:l,:);
    D_knn=D_knn(1:l,:);
    B=B(1:l,:);
end