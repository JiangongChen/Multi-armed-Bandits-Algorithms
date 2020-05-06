% Thompson Sampling algorithm with 2 level feedback
% Input: Number of experiments, number of rounds, mean of first feedback,
% mean of second feedback
% Output: All regrets for each experiment and each round

function all_regrets = TS_2fed(K,T,theta,gamma)
    all_regrets = zeros(K,T);
    miu = theta.*gamma;
    arm_num = length(miu);  % Total number of arms
    % Calculate the optimal arm and its index in order to calculate the regret
    [opt_r,opt_r_index] = max(miu);
    
    % Thompson Sampling algorithm
    for k = 1:K % repeat experiment take average
        
        % Initialize the counters to establish Beta distributions,
        % independent counters for each level feedback
        r_selected = zeros(1, arm_num);
        r_succ_theta = zeros(1, arm_num);
        r_fail_theta = zeros(1, arm_num);

        r_succ_gamma = zeros(1, arm_num);
        r_fail_gamma = zeros(1, arm_num);
        
        round_reward = zeros(1,T);  % Save round reward in each experiment
        
        
        % start experiment
        
        for t = 1:T
            % Generate Beta distributions for each arm according to the counters
            % Draw estimated value from the Beta distributions
            r_rnd_theta = betarnd(r_succ_theta + 1, r_fail_theta + 1);
            r_rnd_gamma = betarnd(r_succ_gamma + 1, r_fail_gamma + 1);
            % Multiple the generated value for each level feedback
            r_rnd = r_rnd_theta.*r_rnd_gamma;
               
            % Choose the largest value
            r_max_index = argmax(r_rnd);
        
            % Observe the reward and update counters
            r_selected(r_max_index) = r_selected(r_max_index) + 1;
            exp_ob_theta = (rand < theta(r_max_index));
            r_succ_theta(r_max_index) = r_succ_theta(r_max_index) + exp_ob_theta;
            r_fail_theta(r_max_index) = r_selected(r_max_index) - r_succ_theta(r_max_index);

            exp_ob_gamma = (rand < gamma(r_max_index));
            r_succ_gamma(r_max_index) = r_succ_gamma(r_max_index) + exp_ob_gamma;
            r_fail_gamma(r_max_index) = r_selected(r_max_index) - r_succ_gamma(r_max_index);
            % Using mean as the round reward
            round_reward(t) = miu(r_max_index);
            
        end
        % Calculate the regrets for each experiment
        all_regrets(k,:) = opt_r*(1:T) - cumsum(round_reward);
    end

end