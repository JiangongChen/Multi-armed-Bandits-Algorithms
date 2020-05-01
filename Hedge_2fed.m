% Hedge algorithm with 2 feedback
% Input: Number of experiments, number of rounds, epsilon, mean of arms feedback 1, mean of arms feedback 2, 
% indicator that determines whether to use dynamic epsilon
% Output: All regrets for each experiment and each round

function all_regrets = Hedge_2fed(K,T,eta,theta,gama,indicator)
    all_regrets = zeros(K,T);
    miu = theta.*gama;  % Overall mean
    arm_num = length(miu);  % Total number of arms
    arm_index = 1:arm_num;  % Used when dealing with more than 2 arms
    % Calculate the optimal arm and its index in order to calculate the regret
    [opt_r,opt_r_index] = max(miu);

    % Hedge algorithm with full information
    for k = 1:K % repeat experiment take average
        % Initialize the losses for each feedback separately
        loss_vector_theta = zeros(1,arm_num); 
        loss_vector_gama = zeros(1,arm_num);

        round_reward = zeros(1,T);  % save round reward in each experiment

        % start experiment

        for t = 1:T
            if indicator 
                eta = sqrt(log(arm_num)/(2*t));
            end
            % Loss vector for each arm
            loss_vector = loss_vector_theta.*loss_vector_gama/t;
            % Calculating weights using exponential, minus a constant to avoid overflow
            weights = exp(-eta*(loss_vector-max(loss_vector)));
            % using softmax to calculate the probability distribution
            sum_weight = sum(weights);
            prob = weights./sum_weight; % The sum of probs must equal to 1

            if arm_num == 2
                % determine which arm we will choose, only for 2 arms
                if rand < prob(1)
                    choice = 1;
                else
                    choice = 2;
                end
            else
                % Choose the arm using probability distributions, for
                % multiple arms
                choice = randsrc(1,1,[arm_index;prob]);
            end

            % Observe the results for all arms, therefore we have full
            % access to information
            for i = 1:arm_num
                exp_ob_theta = (rand < theta(i));
                exp_ob_gama = (rand < gama(i));
                loss_vector_theta(i) = loss_vector_theta(i) + 1 - exp_ob_theta; % Update the losses
                loss_vector_gama(i) = loss_vector_gama(i) + 1 - exp_ob_gama; 
            end

            % Using mean as the round reward
            round_reward(t) = miu(choice);

        end

        % Calculate the regrets for each experiment
        all_regrets(k,:) = opt_r*(1:T) - cumsum(round_reward);
    end
end