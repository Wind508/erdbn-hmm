function batchdata=loaddata(root,n_window)
base_dir=root;
emotions={'angry/','sad/','happy/','disgust/','fear/','surprise/'};
n1=n_window
batchdata=[];
batchdataindex={}; %this is the batch data index, which will give the start point and length
totle_train=0; % the totle number of the validate data
another_test={}; % every cell include a origin data of the aam.
whole_data=[] ;  % save all the data together
for i=1:length(emotions)
    subdir=dir([base_dir emotions{i}]);
    for j=3:length(subdir)
        [base_dir emotions{i} subdir(j).name]
        totle_train=totle_train+1;
        tmp_data=readhtk([base_dir emotions{i} subdir(j).name]);
        start_point=n1+1;
        end_point=size(tmp_data,1)-n1-1;
        video_len=size(tmp_data,1);
        batchdataindex{totle_train}=[start_point end_point video_len];
        another_test{totle_train}=tmp_data;
        tmp_data1=reshape(tmp_data,1,size(tmp_data,1)*size(tmp_data,2));
        whole_data=[whole_data tmp_data1];
    end
end
totle_train
whole_data=zscore(whole_data);
whole_data=reshape(whole_data,length(whole_data)/26,26);
%% deal with all the data
totle_train=0;
frames=1;
for i=1:1%length(emotions)
    subdir=dir([base_dir emotions{i}]);
    for j=3:length(subdir)
        totle_train=totle_train+1
        index=batchdataindex{totle_train};
        tmp_data=whole_data(frames:frames+index(3)-1,:);
        frames=frames+index(3)-1;
        start_point=index(1);
        end_point=index(2);
        for k=start_point:end_point
            tmp_data1=tmp_data(k-n1:k+n1,:);
            [l w]=size(tmp_data1);
            batchdata=[batchdata;reshape(tmp_data1,1,l*w)];
        end
    end
end
end