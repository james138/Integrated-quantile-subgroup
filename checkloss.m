function loss=checkloss(u,tau)
    loss=u.*(tau-(u<0));
end