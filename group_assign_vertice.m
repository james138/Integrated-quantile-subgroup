function [ no_class,class_id ] = group_assign_vertice( V,B,n )
% Calculate cluster assignment from a sequence of ARP solution 

tmp = B(sum(abs(V),1) ==0,:);

% Construct graph with connected edges
G = graph(tmp(:,1)',tmp(:,2)',[],n);
bins = conncomp(G);

class_id = bins;
no_class = max(class_id);

if isempty(tmp)
    class_id = 1:n;
    no_class = n;
end
end

