function y = prox(x, tau,lambda)
    y = x-max((tau-1)/lambda,min(x,tau/lambda));
end