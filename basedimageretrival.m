% This is an implementation of a Content-based Image Retrieval algorithm.
% The execution of the algorithm has been heavily based, but not
% completely,on the article "Multimedia Retrieval through Unsupervised
% Hypergraph-based Manifold Ranking".
% -------------------------------------------------------------------------
% This script file provides fundamental computational functionality in
% order to utilize a pretrained deep convolutional network for image
% retrieval. The key idea behind this approach is to facilated the
% intermediate hidden layer outputs of the neural networks as features, thus
% exploiting the representation power of the deep neural network. The 
% image retrieval task will be computed based on the output of one of the
% last hidden layers of the neural network resnet18.

% The image retrieval task will be applied on a set containing 49 
% images of our own, classified to seven different categories:
% CLASS I:   CAR
% CLASS II:  REMOTE CONTROLLER
% CLASS III: SMARTPHONE
% CLASS IV:  WATER GLASS
% CLASS V:   VIDEO GAME CHARACTER
% CLASS VI:  CUTLERY
% CLASS VII: BOTTLES

% -------------------------------------------------------------------------
% Clear command window and workspace.
% -------------------------------------------------------------------------
clc
clear

% -------------------------------------------------------------------------
% Uncompress the image data folder from the zipped file which already
% exists within the Matlab directory structure. This operation should only 
% be executed only if the subdirectory MerchData does not already exists.
% -------------------------------------------------------------------------
if ~exist('MerchData','dir')
    unzip('MerchData.zip')
end

% -------------------------------------------------------------------------
% Generate the image data store object by parsing the directory structure
% of the loaded data folder. The imageDatastore class is able to
% automatically load the image labels from the names of the corresponding
% sub folders.
% -------------------------------------------------------------------------
image_datastore = imageDatastore('MerchData','IncludeSubfolders',true,...
                                'LabelSource','foldernames');

% Set the percentage of images that will be used for training.
training_percentage = 7;

% Randomly split the generated image data store into traning and testing
% subsets.
[training_datastore,testing_datastore] = splitEachLabel(...
                                                     image_datastore,...
                                                     training_percentage,...
                                                     'randomized');

% -------------------------------------------------------------------------
% Display some random training images.
% -------------------------------------------------------------------------

% Get the number of training images.
TrainImagesNumber = numel(training_datastore.Labels);
% Obtain a random permutation of the training images indices.
PermutationIndices = randperm(TrainImagesNumber);
% Set the number of random training images to be displayed.
DisplayImagesNumber = 49;
% Set the size of the subplot grid.
DisplayGridDimension = sqrt(DisplayImagesNumber);
figure('Name','Training Images Sample');
for image_idx = 1:DisplayImagesNumber
    subplot(DisplayGridDimension,DisplayGridDimension,image_idx);
    I = readimage(training_datastore,PermutationIndices(image_idx));
    image_title = string(training_datastore.Labels(PermutationIndices(image_idx)));
    imshow(I);
    title(image_title);
end
% Set the size of the figure.
set(gcf,'Position',[100 100 800 800])

% -------------------------------------------------------------------------
% Load a pretrained network.
% -------------------------------------------------------------------------

% In this case, ResNet-18 will be in use. ResNet-18 is trained on more than a
% million images and can extract a large number of features.
net = resnet18;

% -------------------------------------------------------------------------
% DO NOT FORGET TO RUN THE FOLLOWING LINE OF CODE IN THE COMMAND LINE!!!!!
% -------------------------------------------------------------------------
% Analyze the network architecture.
% analyzeNetwork(net);
% -------------------------------------------------------------------------

% Get the shape of the image input layer.
% The first layer of the resnet network requires images of size 
% 224 x 224 x 3 where 3 is the number of the color channels.
InputSize = net.Layers(1).InputSize;

% -------------------------------------------------------------------------
% Extract Image Features.
% -------------------------------------------------------------------------

% ResNet requires input images to be of size 224 x 224 x 3. The images 
% contained in the datastore, however, have different sizes. Therefore,
% in order to automatically resize the training and test images, an 
% augmented image datastore should be created by specifying the desired
% image size.
augmented_training_datastore = augmentedImageDatastore(InputSize(1:2),...
                                                       training_datastore);
augmented_testing_datastore = augmentedImageDatastore(InputSize(1:2),...
                                                      testing_datastore);

% The network constructs a hierarchical representation of the input images.
% Deeper layers contain higher-level features, constructed using the
% lower-level features of earlier layers. Getting the feature
% representations of the training and testing images may be accomplished by
% using the activations on the global pooling layer "pool5", at the last 
% layer of the network. The global pooling layer pools the input features
% over all spatial locations, resulting in 512-dimensional feauture vector.

layer = 'pool5';
TrainingFeatures = activations(net,augmented_training_datastore,layer,...
                               'OutputAs','rows');
%TestingFeatures = activations(net,augmented_testing_datastore,layer,...
                              %'OutputAs','rows');

% Get the training and testing labels.
TrainingLabels = training_datastore.Labels;
%TestingLabels = testing_datastore.Labels;
%calculating similarity function
similarity_metric=ones(DisplayImagesNumber,DisplayImagesNumber);
for i=1:DisplayImagesNumber
    for j=1:DisplayImagesNumber
        if(i==j)
            continue
        else
            similarity_metric(i,j) = 1/norm(TrainingFeatures(i,:)-TrainingFeatures(j,:));
        end
    end
end

%iteration process
Wb=zeros(DisplayImagesNumber,DisplayImagesNumber);
condition=220;
while condition>10
%for objects
T=zeros(DisplayImagesNumber,DisplayImagesNumber);
%for positions
Tq=zeros(DisplayImagesNumber,DisplayImagesNumber);
for k=1:DisplayImagesNumber
    [~,T(k,:)] = sort(similarity_metric(k,:),'descend');
    r = 1:DisplayImagesNumber;
    r(T(k,:)) = r; 
    Tq(k,:) = r;
end
%let L be L<<DisplayImagesNumber such that CL ⊂ C
L=10;
%let N be L-nearest neighbours
N=Tq(:,1:L);
%A)Rank Normalization

rank=zeros(DisplayImagesNumber:L);
for i = 1:DisplayImagesNumber
    for j=1:DisplayImagesNumber
        rank(i,j)=(Tq(i,j)+Tq(j,i))/2;
    end
end
%B)Hypergraph construction
%Vi Training features(i,:)
%E:Hyperedges 
%We assume E=N where L=5
%Hb incidence matrix 1 and 1
Hb=zeros(DisplayImagesNumber,DisplayImagesNumber);
%Weight function
index1=1;
for i=1:DisplayImagesNumber
    for j=1:L
        index1=N(i,j);
        Hb(i,index1)=1;
    end
end
%H continuous incidence matrix
H=zeros(DisplayImagesNumber,DisplayImagesNumber);
%Weight function
for i=1:DisplayImagesNumber
    for j=1:DisplayImagesNumber
        %index=N(i,j);
        %H(i,index)=1-(log(Tq(i,index))./log(L+1));
        H(i,j)=1-(log(Tq(i,j))./log(L+1));
        if (H(i,j)<0)
            H(i,j)=0;
        end
    end
end


%C) Hyperedge similarities
Sh=H*H.';
Su=H.'*H;
S=Sh.*Su;
%D)Cartesian product of hyperedge elements
%eq^2 dimension of cartesian product of each hyperedge
%not needed
%{
Esquare=zeros(L,L,DisplayImagesNumber);
for k=1:DisplayImagesNumber
    for i=1:L
        for j=1:L
           Esquare(i,j,k)=N(k,i)*N(k,j);
        end
    end
end
%}
%w(ei)
Wsum=zeros(DisplayImagesNumber,1);
for k=1:DisplayImagesNumber
    Wsum(k)=sum(H(k,:));
end
%p: pairwise similarity matrix p:ExVxV->R^+
p=zeros(DisplayImagesNumber,DisplayImagesNumber,DisplayImagesNumber);
for eq=1:DisplayImagesNumber
    for i=1:DisplayImagesNumber
        %ui=N(eq,i);  %we'll need the possition of the vertex ui in the E 
                     %matrix, because H matrix is sparce
        for j=1:DisplayImagesNumber
            %uj=N(eq,j);
            p(i,j,eq)=Wsum(eq)*H(eq,i)*H(eq,j);
        end
    end
end
%calculating matrix C
C=zeros(DisplayImagesNumber,DisplayImagesNumber);
for ui=1:DisplayImagesNumber
    for uj=1:DisplayImagesNumber
       Csum=0;
       for eq=1:DisplayImagesNumber
           Csum = Csum + p(ui,uj,eq);
       end
       C(ui,uj)=Csum;
    end
end
%E) Hypergraph base similarity
W=C.*S;
condition=norm(W-Wb);
Wb=W;
for i=1:DisplayImagesNumber
        W(i,i)=1000000;
end
similarity_metric=W;
end
%Final computation of Tq 
for k=1:DisplayImagesNumber
    [~,T(k,:)] = sort(similarity_metric(k,:),'descend');
    r = 1:DisplayImagesNumber;
    r(T(k,:)) = r; 
    Tq(k,:) = r;
end   
%accuracy 
accuracy=0;
%Presentation of the results
figure('Name','Result');
for k = 1:7
    subplot(7,L+1,(k-1)*10+k);
    I = readimage(training_datastore,k);
    
    imshow(I);
    title('Target image');
    for o=2:L+1
        subplot(7,L+1,(k-1)*10 +k -1+ o);
        Is = readimage(training_datastore,T(k,o-1));
        imshow(Is);
        image_title = string(training_datastore.Labels(T(k,o-1)));
        title(image_title);
        if (image_title==string(training_datastore.Labels(k)))
            accuracy=accuracy+1;
        end
    end
    
end
set(gcf,'Position',[0 0 2800 1730])%scaled output based on screen resolution
figure('Name','Result');
for k = 1:7
    subplot(7,L+1,(k-1)*10+k);
    I = readimage(training_datastore,k+7);
   
    imshow(I);
    title('Target image');
    for o=2:L+1
        subplot(7,L+1,(k-1)*10 +k -1+ o);
        Is = readimage(training_datastore,T(k+7,o-1));
        imshow(Is);
        image_title = string(training_datastore.Labels(T(k+7,o-1)));
        title(image_title);
        if (image_title==string(training_datastore.Labels(k+7)))
            accuracy=accuracy+1;
        end
    end
    
end  
set(gcf,'Position',[0 0 2800 1730])
figure('Name','Result');
for k = 1:7
    subplot(7,L+1,(k-1)*10+k);
    I = readimage(training_datastore,k+14);
   
    imshow(I);
    title('Target image');
    for o=2:L+1
        subplot(7,L+1,(k-1)*10 +k -1+ o);
        Is = readimage(training_datastore,T(k+14,o-1));
        imshow(Is);
        image_title = string(training_datastore.Labels(T(k+14,o-1)));
        title(image_title);
        if (image_title==string(training_datastore.Labels(k+14)))
            accuracy=accuracy+1;
        end
    end
    
end  
set(gcf,'Position',[0 0 2800 1730])
figure('Name','Result');
for k = 1:7
    subplot(7,L+1,(k-1)*10+k);
    I = readimage(training_datastore,k+21);
   
    imshow(I);
    title('Target image');
    for o=2:L+1
        subplot(7,L+1,(k-1)*10 +k -1+ o);
        Is = readimage(training_datastore,T(k+21,o-1));
        imshow(Is);
        image_title = string(training_datastore.Labels(T(k+21,o-1)));
        title(image_title);
        if (image_title==string(training_datastore.Labels(k+21)))
            accuracy=accuracy+1;
        end
    end
    
end  
set(gcf,'Position',[0 0 2800 1730])
figure('Name','Result');
for k = 1:7
    subplot(7,L+1,(k-1)*10+k);
    I = readimage(training_datastore,k+28);
   
    imshow(I);
    title('Target image');
    for o=2:L+1
        subplot(7,L+1,(k-1)*10 +k -1+ o);
        Is = readimage(training_datastore,T(k+28,o-1));
        imshow(Is);
        image_title = string(training_datastore.Labels(T(k+28,o-1)));
        title(image_title);
        if (image_title==string(training_datastore.Labels(k+28)))
            accuracy=accuracy+1;
        end
    end
    
end  
set(gcf,'Position',[0 0 2800 1730])
figure('Name','Result');
for k = 1:7
    subplot(7,L+1,(k-1)*10+k);
    I = readimage(training_datastore,k+35);
   
    imshow(I);
    title('Target image');
    for o=2:L+1
        subplot(7,L+1,(k-1)*10 +k -1+ o);
        Is = readimage(training_datastore,T(k+35,o-1));
        imshow(Is);
        image_title = string(training_datastore.Labels(T(k+35,o-1)));
        title(image_title);
        if (image_title==string(training_datastore.Labels(k+35)))
            accuracy=accuracy+1;
        end
    end
    
end  
set(gcf,'Position',[0 0 2800 1730])
figure('Name','Result');
for k = 1:7
    subplot(7,L+1,(k-1)*10+k);
    I = readimage(training_datastore,k+42);
   
    imshow(I);
    title('Target image');
    for o=2:L+1
        subplot(7,L+1,(k-1)*10 +k -1+ o);
        Is = readimage(training_datastore,T(k+42,o-1));
        imshow(Is);
        image_title = string(training_datastore.Labels(T(k+42,o-1)));
        title(image_title);
        if (image_title==string(training_datastore.Labels(k+42)))
            accuracy=accuracy+1;
        end
    end
    
end  
set(gcf,'Position',[0 0 2800 1730])
%total accuracy percentage
accuracy_percentage=(accuracy/(DisplayImagesNumber*7))*100;









