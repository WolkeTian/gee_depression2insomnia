function CPM_Results = cpm_tian(fnc_mats,behav_vector,thresh_set, fold, isDirected)
    % connectome-based predictive model
    % doi.org/10.1038/nprot.2016.178
    % fnc_mats: subs * nodes * nodes
    % isDirected: 1, pos and neg seperately to predict; 0, together.
    % start
    
    % threshold for feature selection
    thresh = thresh_set;
    
    % ---------------------------------------
    sub_nums = size(fnc_mats,1);
    node_nums = size(fnc_mats,2);
    % initial prediction scores to store
    [behav_pred_pos, behav_pred_neg] = deal(zeros(sub_nums,1));

    % fncmat to 2d
    % generate N*M*M logical index only tril
    tril_index = logical(tril(ones(node_nums), -1));
    tril_indexs = permute(repmat(tril_index, [1,1, sub_nums]), [3,1,2]);
    fnc2d = reshape(fnc_mats(tril_indexs), sub_nums, []); % to N * edges

	k = fold;
    
    predict_behav = zeros(numel(behav_vector), 1);
    predict_behav = predict_behav(1:floor(sub_nums/k) *k); % k-fold后，不能整除的尾部剔除掉
    [behav_pred_pos, behav_pred_neg] = deal(predict_behav); % 生成正负两个供后续可能使�?
    % initial edge mask sets, k fold to store k*M*M mat
    [mask_pos_set, mask_neg_set] = deal(zeros(k, node_nums, node_nums));
    
    issigmoidal = 1; % 默认不启用sigmoidal函数
    if issigmoidal disp('Sigmoidal function is actived'); end
    for i = 1:k
        
        leftout = i:k:sub_nums; %抛出作为验证集的
        leftout = leftout(1:floor(sub_nums/k)); % 维持长度为整除项 且一�?
        % fprintf('\n Leaving out subject # %6.3f',leftout); % 输出交叉验证抛出被试信息
        % leave out subject from matrices and behavior
        train_fnc = fnc2d; train_fnc(leftout,:) = [];
        test_fnc = fnc2d(leftout, :);
        
        train_behav = behav_vector; train_behav(leftout) = [];
        test_behav = behav_vector(leftout);

        % correlate all edges with behavior

        [r_mat,p_mat] = corr(train_fnc,train_behav); %计算抛出�?个被试后的r和p，未校正
        % [r_mat,p_mat] = corr(train_fnc,train_behav, 'Type', 'Spearman'); % 斯皮尔曼等级相关
        r_mat(isnan(r_mat))= 0;
        p_mat(isnan(p_mat))= 0; % nan变成0

        % set threshold and define masks

        pos_mask = zeros(size(train_fnc,2),1); % edges数字*1的向�?
        neg_mask = zeros(size(train_fnc,2),1);


        pos_edges = find(r_mat > 0 & p_mat < thresh); % 找到p小于阈�?�的正负连接
        neg_edges = find(r_mat < 0 & p_mat < thresh);

        pos_mask(pos_edges) = 1; % 根据上边的，制作mask
        neg_mask(neg_edges) = 1;
        
        if issigmoidal
            %----------------------------alternative sigmoidal 加权--------------
            % 转换p阈�?�到r阈�??            
            T_thre = tinv(thresh/2, numel(train_behav) - 2); %Student's t inverse cumulative distribution function
            R_thre = sqrt(T_thre^2/(numel(train_behav) - 2 + T_thre^2)); %得到的R阈�??
                    
            % 用sigmoidal函数创建加权mask
            % 当相�?=R/3时，weight = 0.5; 当相�?=R时，weight = 0.88, R越大权重越大
            pos_mask(pos_edges) = sigmf(r_mat(pos_edges), [3/R_thre, R_thre/3]);
            neg_mask(neg_edges) = sigmf(r_mat(neg_edges), [-3/R_thre, R_thre/3]);
        end

        
        % transfer mask to M*M
        pos_mask2d = zeros(node_nums); pos_mask2d(tril_index) = pos_mask;
        neg_mask2d = zeros(node_nums); neg_mask2d(tril_index) = neg_mask;

        mask_pos_set(i, :, :) = pos_mask2d; % to store mask sets
        mask_neg_set(i, :, :) = neg_mask2d;
        
        % get sum of all edges in TRAIN subs (divide by 2 to control for the
        % fact that matrices are symmetric)
        sum_pos_links = train_fnc * pos_mask; % sum of pos_links
        
        sum_neg_links = train_fnc * neg_mask; % sum of neg_links
        
        test_sumpos = test_fnc * pos_mask;
        test_sumneg = test_fnc * neg_mask;
        
        if isDirected == 1
            % build model on TRAIN subs
            poly_pos = polyfit(sum_pos_links, train_behav,6); % 用训练被试显著相关的连接值的和，预测行为，线性拟�?
            poly_neg = polyfit(sum_neg_links, train_behav,6);
            % 模型，y = a*x + b;得到两个系数，a和b
            % run model on TEST sub
            
            behav_pred_pos(leftout) = polyval(poly_pos, test_sumpos);
            behav_pred_neg(leftout) = polyval(poly_neg, test_sumneg);% 得到这层循环验证集用正\负连接的预测�?
            
            %behav_pred_pos(leftout) = coef_pos(1)*test_sumpos + coef_pos(2);
            %behav_pred_neg(leftout) = coef_neg(1)*test_sumneg + coef_neg(2); 
        elseif isDirected == 0
            % 正负连接共同预测
            % coef = polyfit([sum_pos_links, sum_neg_links], train_behav,1); 
            coef = regress(train_behav,[sum_pos_links, sum_neg_links, ones(numel(sum_neg_links), 1)]);
            
            % 进行验证
            predict_behav(leftout) = coef(1) * test_sumpos + coef(2) * test_sumneg + coef(3);
            
%             % 不用加权和，用全部单个连接信�?
%             coefs = regress(train_behav,[train_fnc(:, pos_edges), train_fnc(:, neg_edges), ones(size(train_fnc, 1), 1)]);
%             % 进行验证
%             predict_behav(leftout) = [test_fnc(:, pos_edges), test_fnc(:, neg_edges), ones(size(test_fnc, 1), 1)] * coefs;
        end
    end
    % 循环结束
    
    if isDirected == 1
        % �?有被试都作为验证集，得到预测�?
        value2pred = behav_vector; value2pred = value2pred(1:numel(predict_behav)); % matche length

        value2pred((isnan(behav_pred_pos)))=[]; % 预测值是空�?�的被扔�?
        behav_pred_pos(isnan(behav_pred_pos))=[];

        [R_pos, P_pos] = corr(behav_pred_pos,value2pred); % �?测正连接预测效果

        value2pred_2 = behav_vector; value2pred_2 = value2pred_2(1:numel(predict_behav));
        
        value2pred_2((isnan(behav_pred_neg)))=[];
        behav_pred_neg(isnan(behav_pred_neg))=[];
        [R_neg, P_neg] = corr(behav_pred_neg,value2pred_2);  % �?测负连接预测效果
        % compare predicted and observed scores

        
        % 得到用正/负相关分别预测的表现

        CPM_Results.R_pos = R_pos;
        CPM_Results.P_pos = P_pos;
        CPM_Results.pos_predict = behav_pred_pos; % 存预测�??
        CPM_Results.pos_test = value2pred; %原始行为成绩
        % 存储mask，如果用了sigm函数，则权重矩阵转为2值矩�?
        CPM_Results.mask_pos = mask_pos_set ~= 0;


        CPM_Results.R_neg = R_neg;
        CPM_Results.P_neg = P_neg;
        CPM_Results.neg_predict = behav_pred_neg;
        CPM_Results.mask_neg = mask_neg_set ~= 0;
        % pos figure
        figure(1); plot(value2pred,behav_pred_pos,'r.'); 
        lsline;title(['pos: p = ', num2str(P_pos),'|', 'r = ', num2str(R_pos)])
        % neg figure
        figure(2); plot(value2pred_2,behav_pred_neg,'b.'); 
        lsline;title(['neg: p = ', num2str(P_neg),'|','r = ', num2str(R_neg)]);
        
        
    elseif isDirected == 0 % 正负连接�?起预�?
        value2pred = behav_vector; value2pred = value2pred(1:numel(predict_behav)); % matche length

        value2pred((isnan(predict_behav)))=[]; % 预测值是空�?�的被扔�?
        predict_behav(isnan(predict_behav))=[];
        [R, P] = corr(predict_behav,value2pred); % �?测正连接预测效果
        % [R, P] = corr(predict_behav,value2pred, 'Type', 'Spearman'); % 斯皮尔曼等级相关
        
        % permutation得到r值分�?
        permtimes = 1e4;
        r_dist = zeros(1,1e4);
        disp(['start to permutation test for ', num2str(permtimes), 'please wait']);
        for i = 1:permtimes
            permed_pred = predict_behav(randperm(numel(predict_behav)));
            r_dist(i) = corr(permed_pred, value2pred);            
        end
        P_permed = sum(abs(r_dist)> abs(R))/1e4;
%         % one-side p-value
%         if R >0
%             P_permed = sum(r_dist> R)/1e4;
%         else
%             P_permed = sum(r_dist< R)/1e4;            
%         end
        % compare predicted and observed scores
        % 得到用正/负相关分别预测的表现
        CPM_Results.R = R;
        CPM_Results.P = P;
        CPM_Results.P_permtest = P_permed;
        CPM_Results.predict = predict_behav; % 存预测�??
        CPM_Results.topredict = value2pred; %原始行为成绩
        
        % 存储mask，如果用了sigm函数，则权重矩阵转为2值矩�?
        CPM_Results.mask_pos = mask_pos_set ~= 0;
        CPM_Results.mask_neg = mask_neg_set ~= 0; 
        
        %figure
        figure(1); plot(value2pred,predict_behav,'r.'); 
        lsline;title(['p = ', num2str(P),'|', 'r = ', num2str(R)]);
    end
    
    % find stable connectivity
    criteria = 0.9;% 90% default could be
    disp(['The default criteria for store valuable connectivity is ', num2str(criteria * 100), '%']);
    CPM_Results.stable_poslinks = squeeze(mean(CPM_Results.mask_pos, 1)) >= criteria;
    CPM_Results.stable_neglinks = squeeze(mean(CPM_Results.mask_neg, 1)) >= criteria;
    
    % function is over
end