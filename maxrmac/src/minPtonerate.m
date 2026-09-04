% function [f, b, e] = minPtonerate(H, theta, w, cb)
% minPtone maximizes f = sum_{u=1}^U lambda_u * b_u - sum_{u=1}^U w_u * e_u 
% subject to b \in C_g(H,e)
%
% the inputs are:
% 1)  H, an Ly by U channel matrix. Ly is the number of receiver antennas, 
%     U is the total number of users.
%     H(:,u) is the channel for user u. In this code we assume each user 
%     only has single transmit antenna, thus H(:,u) is a column vector. 
% 2)  theta, a U by 1 vector containing the weights for the rates.
% 3)  w, a U by 1 vector containing the weights for each user's energy.
% 4)  cb=1 for complex bb and cb=2 for real bb
%
% the outputs are:
% 1)  f, the minimum value (or maximum value of the -1 * function).
% 2)  b, a U by 1 vector containing the rates for all users
%     that optimizes the given function.
% 3)  e, a U by 1 vector containing the energies for all users 
%     that optimizes the given function.
function [f, b, e] = minPtonerate(H, theta, w, cb)

[Ly, U] = size(H);
[stheta, ind] = sort(-theta);
stheta = -stheta;
sH = H(:,ind);
sw = w(ind);

NT_max_it = 1000;                              % Maximum number of Newton's 
                                               % method iterations
dual_gap = 1e-6;
mu = 10;                                       % step size for t
alpha = 0.01;                                  % back tracking line search parameters
beta = 0.5;

count = 1;
nerr = 1e-5;                                   % acceptable error for inner loop 
                                               % Newton's method

e = ones(U,1);                                 % Strictly feasible point;

t = .1;                                         
l_p = 1;                                       % for newton's method termination



while (1+U)/t > dual_gap
   t =  t * mu;
   l_p = 1;
   count = 1;
   while l_p > nerr & count < NT_max_it 
      
      f_val = eval_f(t * stheta, sH, e, t * sw);         
                                               % calculating function value
      
                                               % calculating the hessian and gradient      
      [g, h] = Hessian(t * stheta, sH, e, t * sw);
         
      de = -real(h\g);                               % search direction
      
      l_p = g' * de;                           % theta(e)^2 for Newton's method
      
      s = 1;                                   % checking e = e+s*de feasible 
                                               % and also back tracking algorithm
   
      e_new = e + s * de;
      
     
      
      if e_new > zeros(U,1)
          f_new = eval_f(t * stheta, sH, e_new, t * sw);
          feas_check = (f_new > f_val + alpha * s * g' * de);
      else
          feas_check = 0;
      end


            
       
   
      while ~feas_check                
         s = s * beta;
         if s < 1e-40
             l_p = nerr/2;
             break
         end
         e_new = e + s * de;
         
         if e_new > zeros(U,1)
             f_new = eval_f(t * stheta, sH, e_new, t * sw);
             feas_check = (f_new > f_val + alpha * s * g' * de);
         else
             feas_check = 0;
         end

      end   
      
      e = e + s * de;                         % update e
      count = count + 1;                      % number of Newtons method iterations
   end
  
end


M = eye(Ly) + sH(:,1) * sH(:,1)' * e(1);
b = zeros(U,1);
b(1) =  (1/cb) * real(log(det(M)));
for u = 2:U
    b(u) = -(1/cb) * real(log(det(M)));
    M = M + sH(:,u) * sH(:,u)' * e(u);
    b(u) = b(u) + (1/cb) * real(log(det(M)));
end


b(ind) = b;
e(ind) = e;
f = theta' * b - w' * e; 
