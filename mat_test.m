clear;
clc;

n = 5;
rho = 5;

n_sim = 1000;
n_architecture = 1;
B_set = eye(n);

ctrb_rank_count = 0;
ctrb_gain_count = 0;
ctrb_dare_mat_check = 0;

Q = eye(n);
R = eye(n_architecture);

for t = 1:n_sim
    disp(strcat("t=", num2str(t)))
    Adj = rand(n,n);
    A = rho*Adj/max(abs(eig(Adj)));
    B_active = randperm(n, n_architecture);
    B = zeros(n, n_architecture);
    for j = 1:length(B_active)
        B(1:n, j) = B_set(1:n, B_active(j));
    end
    if rank(ctrb(A,B)) == n
        ctrb_rank_count = ctrb_rank_count + 1;
    end

    [X,K,L] = idare(A,B,Q,R,[],[]);
    if min(eig(X)) >= 0
        ctrb_dare_mat_check = ctrb_dare_mat_check + 1;
    end
    if max(abs(L)) <= 1
        ctrb_gain_count = ctrb_gain_count + 1;
    end
end

disp(['Ctrb matrix fail:', num2str(n_sim - ctrb_rank_count)])
disp(['DARE cost mat fail:', num2str(n_sim - ctrb_dare_mat_check)])
disp(['DARE gain fail:', num2str(n_sim - ctrb_gain_count)])