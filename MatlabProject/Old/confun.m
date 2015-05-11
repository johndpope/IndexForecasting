function [c, ceq] = confun(w, covar, target_risk)
PortfolioRisk    = @(w,cov) (w*cov*w');

% Nonlinear inequality constraints
c   =  [w(1) + w(2) - 1 ;
       -w'];
   

% Nonlinear equality constraints
ceq = [w(3) + w(4) - 1 ;
       sqrt(PortfolioRisk(w,covar)) - target_risk];
end