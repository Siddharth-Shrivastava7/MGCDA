



function result_info=my_gen_seg_result(seg_param, ds_info, predict_info)
    
    result_info=[];
    
    img_size=seg_param.img_size_input;
       
    gt_mask=seg_param.gt_mask_input;
    assert(~isempty(gt_mask));
    gt_mask=uint8(gt_mask);
    assert(all(img_size==size(gt_mask)));
        
    
    eva_param=seg_param.eva_param;
    class_info=eva_param.class_info;
            
    assert(~isempty(predict_info));
    
    score_map=single(predict_info.score_map);
    score_map_org=score_map;
    
    score_map_size=size(score_map);
    score_map_size=score_map_size(1:2);
        
    if any(img_size~=score_map_size)
        score_map=log(score_map);
        score_map=max(score_map, -20);
        score_map=my_resize(score_map, img_size);
        score_map=exp(score_map);
    end
       

    [~, predict_mask]=max(score_map,[],3);
    predict_mask=uint8(gather(predict_mask));
    
    [aa, indices] = sort(score_map,3,'descend');
    predict_mask2 = uint8(gather(indices(:,:,2)));

    exclude_class_idxes=class_info.void_class_idxes;
    class_num=class_info.class_num;
    assert(all(size(gt_mask)==size(predict_mask)));
    assert(all(size(gt_mask)==size(predict_mask2)));
    gt_mask_vector=gt_mask(:);
    predict_mask_vector=predict_mask(:);
    predict_mask_vector2=predict_mask2(:);
    if ~isempty(exclude_class_idxes)
        valid_pixel_sel=~ismember(gt_mask, exclude_class_idxes);
        gt_mask_vector=gt_mask_vector(valid_pixel_sel);
        predict_mask_vector=predict_mask_vector(valid_pixel_sel);
        predict_mask_vector2=predict_mask_vector2(valid_pixel_sel);
    end
    [tmp_con_mat, label_vs]=confusionmat(gt_mask_vector, predict_mask_vector);
    [tmp_con_mat2, label_vs2]=confusionmat(gt_mask_vector, predict_mask_vector2);
    confusion_mat=zeros(class_num, class_num);
    confusion_mat2=zeros(class_num, class_num);
    confusion_mat(label_vs,label_vs)=tmp_con_mat;
    confusion_mat2(label_vs2,label_vs2)=tmp_con_mat2;
    accuracy_classes=zeros(class_num, 1);
    
    valid_gt_class_sel=true(class_num, 1);
    acc_result_info=[];

    for c_idx=1:class_num
        % one_true_pos_num=confusion_mat(c_idx,c_idx) + confusion_mat2(c_idx,c_idx);
        one_true_pos_num=confusion_mat(c_idx,c_idx);
        one_gt_pos_num=sum(confusion_mat(c_idx,:));

        if one_gt_pos_num>0
                one_accuracy=one_true_pos_num/(one_gt_pos_num+eps);
                accuracy_classes(c_idx)=one_accuracy;
                
        else
            valid_gt_class_sel(c_idx)=false;
        end
    end
    valid_class_sel=true(class_num, 1);
    valid_class_sel(exclude_class_idxes)=false;
    valid_class_sel=valid_class_sel & valid_gt_class_sel;
    accuracy_per_class=mean(accuracy_classes(valid_class_sel));
    acc_result_info.accuracy_classes = accuracy_classes(valid_class_sel);
    acc_result_info.accuracy_per_class = accuracy_per_class;
    img_idx=seg_param.img_idx;
    img_name=ds_info.img_names{img_idx};

    save([img_name '.mat'],'-struct','acc_result_info');

        
    one_seg_eva_result=seg_eva_one_img(predict_mask, gt_mask, class_info);
    result_info.seg_eva_result=one_seg_eva_result;
       
    coarse_gt_mask=imresize(gt_mask, score_map_size, 'nearest');
    [~, coarse_predict_mask]=max(score_map_org,[],3);
    coarse_predict_mask=uint8(gather(coarse_predict_mask));
    one_seg_eva_result_coarse=seg_eva_one_img(coarse_predict_mask, coarse_gt_mask, class_info);
    result_info.seg_eva_result_coarse=one_seg_eva_result_coarse;
        
    
    predict_result_densecrf=[];
    result_info.seg_eva_result_densecrf=[];
    
    if eva_param.eva_densecrf_postprocess
        task_config=[];
        task_config.img_data=seg_param.img_data_input;
        task_config.score_map=score_map;
        predict_result_densecrf=gen_prediction_densecrf(task_config, eva_param);
        one_seg_eva_result_densecrf=seg_eva_one_img(predict_result_densecrf.predict_mask, gt_mask, class_info);
        result_info.seg_eva_result_densecrf=one_seg_eva_result_densecrf;
    end
       
    
    do_save_results(ds_info, seg_param, predict_mask, score_map_org, predict_result_densecrf);
      
end



function do_save_results(ds_info, seg_param, predict_mask_net, score_map_org, predict_result_densecrf)

    img_idx=seg_param.img_idx;
    eva_param=seg_param.eva_param;
    class_info=eva_param.class_info;
    
    predict_mask_data=class_info.class_label_values(predict_mask_net);      
    assert(isa(class_info.class_label_values, 'uint8'));	
    assert(isa(predict_mask_data, 'uint8'));
    
    img_name=ds_info.img_names{img_idx};
    
    if eva_param.save_predict_mask
        tmp_dir=eva_param.predict_result_dir_mask;
        mkdir_notexist(tmp_dir);
        one_cache_file=fullfile(tmp_dir, [img_name '.png']);
        imwrite(predict_mask_data, class_info.mask_cmap, one_cache_file);
        
    end
    
    if eva_param.save_predict_result_full
            
        % notes: saved score map values range from 0 to 255
        save_full_prediction_as_single_in_cpu =...
            contains(ds_info.ds_name, 'Dark_Zurich_test')...
            || contains(ds_info.ds_name, 'Dark_Zurich_val')...
            || contains(ds_info.ds_name, 'Dark_Zurich_twilight')...
            || contains(ds_info.ds_name, 'Dark_Zurich_night');
        % Modification: save score map in original single format.
        if ~save_full_prediction_as_single_in_cpu
            score_map_org=im2uint8(score_map_org);
        end
        tmp_dir=eva_param.predict_result_dir_full;
        one_cache_file=fullfile(tmp_dir, [img_name '.mat']);
        tmp_result_info=[];
        tmp_result_info.mask_data=predict_mask_data;
        if ~save_full_prediction_as_single_in_cpu
            tmp_result_info.score_map=score_map_org;
        else
            % Modification: save score map as a CPU array.
            tmp_result_info.score_map=gather(score_map_org);
        end
        tmp_result_info.img_size=size(predict_mask_data);
        tmp_result_info.class_info=class_info;
        my_save_file(one_cache_file, tmp_result_info, true, true);    		
    end
    
       
    
    if ~isempty(predict_result_densecrf)

    	assert(eva_param.predict_save_mask)
        tmp_dir=eva_param.predict_result_dir_densecrf;
        mkdir_notexist(tmp_dir);
        save_mask_data=class_info.class_label_values(predict_result_densecrf.predict_mask);      
		assert(isa(save_mask_data, 'uint8'));
        one_cache_file=fullfile(tmp_dir, [img_name '.png']);
        imwrite(save_mask_data, class_info.mask_cmap, one_cache_file);
   
    end
    
    
end
