function result=KPplusQinBIC(E,beta,pplusq,k)
    q_hat=pplusq-sum(all(round(E*beta,4)));
    if q_hat==pplusq
        result=pplusq;
    else
        result=k*(pplusq-q_hat)+q_hat;
    end
end
