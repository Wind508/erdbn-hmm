function [model output]=rbm(batchdata,numhid,maxepoch)
% this is just a simple binary-binary rbm 

epsilonw      = 0.05;   % Learning rate for weights
epsilonvb     = 0.05;   % Learning rate for biases of visible units
epsilonhb     = 0.05;   % Learning rate for biases of hidden units

CD=1;
weightcost  = 0.001;
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases numdims numbatches]=size(batchdata);
epoch=1;

% Initializing symmetric weights and biases.
vishid     = 0.001*randn(numdims, numhid);
hidbiases  = zeros(1,numhid);
visbiases  = zeros(1,numdims);

model.vishid=vishid;
model.hidbiases=hidbiases;
model.visbiases=visbiases;

poshidprobs = zeros(numcases,numhid);
neghidprobs = zeros(numcases,numhid);
posprods    = zeros(numdims,numhid);
negprods    = zeros(numdims,numhid);
vishidinc  = zeros(numdims,numhid);
hidbiasinc = zeros(1,numhid);
visbiasinc = zeros(1,numdims);
batchposhidprobs=zeros(numcases,numhid,numbatches);

for epoch = epoch:maxepoch
    fprintf(1,'epoch %d\r',epoch);
    errsum=0;
    for batch = 1:numbatches,
%         fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        visbias = repmat(visbiases,numcases,1);
        hidbias = repmat(2*hidbiases,numcases,1);
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        data = data > rand(numcases,numdims);
        
        poshidprobs = 1./(1 + exp(-data*(2*vishid) - hidbias));
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        negdata = 1./(1 + exp(-poshidstates*vishid' - visbias));
        negdata = negdata > rand(numcases,numdims);
        neghidprobs = 1./(1 + exp(-negdata*(2*vishid) - hidbias));
        
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;
        
        if epoch>5,
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = momentum*vishidinc + ...
            epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
        
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
    
end;
output=batchposhidprobs;
save fullmnistvh vishid visbiases hidbiases epoch

end
