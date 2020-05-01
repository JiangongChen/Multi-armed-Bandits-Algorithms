function choice = argmax(vector)
    top = -inf;
    ties = [];
    index = 1:length(vector);
    for i = 1:length(vector)
        if vector(i)>top
            top = vector(i);
            ties = [];
        end
        if vector(i)==top
            ties=[ties i];
        end
    end
    if length(ties)==1
        choice = ties;
    else
        choice = randsample(index(ties),1);
    end
end