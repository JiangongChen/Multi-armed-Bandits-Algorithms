% UCB algorithm
% Input: Number of experiments, number of rounds, alpha, mean of arms
% Output: All regrets for each experiment and each round

function all_regrets = UCB(K,T,alp,miu)
    all_regrets = zeros(K,T);
    arm_num = length(miu);  % Total number of arms
    % Calculate the optimal arm and its index in order to calculate the regret
    [opt_r,opt_r_index] = max(miu);
    
    % UCB algorithm
    for k = 1:K % repeat experiment take average
        
        % Initialize the counters to estimate rewards for each arm
        r_selected = zeros(1, arm_num);
        r_succ = zeros(1, arm_num);
        r_fail = zeros(1, arm_num);
        
        round_reward = zeros(1,T);  % Save round reward in each experiment
        
        
        % start experiment
        
        for t = 1:T
            % In first 'arm_num' rounds, choose each arm once
            if t <= arm_num
                r_max_index = t;
            else
                r_est = r_succ ./ r_selected;
                % Using upper confidence bound to calculate the estimated
                % reward
                r_hat = r_est + sqrt(alp*log(t)./(2*r_selected));
                %r_hat = min(r_est + sqrt(alp*log(t)./(2*r_selected)),1);
                [r_max,r_max_index] = max(r_hat);
            end
            % Observe the reward and update counters
            exp_ob = (rand < miu(r_max_index));
            r_selected(r_max_index) = r_selected(r_max_index) + 1;
            r_succ(r_max_index) = r_succ(r_max_index) + exp_ob;
            r_fail(r_max_index) = r_selected(r_max_index) - r_succ(r_max_index);
            % Using mean as the round reward
            round_reward(t) = miu(r_max_index);
            
        end
        % Calculate the regrets for each experiment
        all_regrets(k,:) = opt_r*(1:T) - cumsum(round_reward);
    end
end