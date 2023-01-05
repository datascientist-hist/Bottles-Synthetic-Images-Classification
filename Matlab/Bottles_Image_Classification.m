path_to_images = "C:\Users\pepee\Desktop\Università\2 Anno\Neural Computing\Bottles Image Classification\MatLab\Bottle_Images";

image_datastore = imageDatastore(path_to_images, "IncludeSubfolders",true,"LabelSource","foldernames");

[train,validation,test] = splitEachLabel(image_datastore,0.1, 0.04, 0.86, 'randomized');

%%
rsz_train=augmentedImageDatastore([224 224 3],train);
rsz_validation = augmentedImageDatastore([224 224 3],validation);
rsz_test = augmentedImageDatastore([224 224 3],test);
%%
opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.01,...
    "MaxEpochs",10,...
    "MiniBatchSize",64,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",70,...
    "Plots","training-progress",...
    "ValidationData",rsz_validation,...
    "Momentum",0.9);

[net, traininfo] = trainNetwork(rsz_train,resnet_18,opts);
%%
true_validation_labels = validation.Labels;
[pred_validation_labels,score_validation] = classify(net, rsz_validation);
accuracy_validation = mean(true_validation_labels == pred_validation_labels);
ConfVal = confusionmat(true_validation_labels, pred_validation_labels);
confusionchart(ConfVal)


%%
true_test_labels = test.Labels;
[pred_test_labels,score_test] = classify(net, rsz_test);
accuracy_test = mean(true_test_labels == pred_test_labels);
C = confusionmat(true_test_labels, pred_test_labels);
confusionchart(C)


%%
chosenClass = "Water Bottle";
classIdx = find(net.Layers(end).Classes == chosenClass);

numImgsToShow = 9;

[sortedScores,imgIdx] = findMinActivatingImages(test,chosenClass,score_test,numImgsToShow);


figure
plotImages(test,imgIdx,sortedScores,pred_test_labels,numImgsToShow)
%%
function [sortedScores,imgIdx] = findMaxActivatingImages(imds,className,predictedScores,numImgsToShow)
% Find the predicted scores of the chosen class on all the images of the chosen class
% (e.g. predicted scores for sushi on all the images of sushi)
[scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores);

% Sort the scores in descending order
[sortedScores,idx] = sort(scoresForChosenClass,'descend');

% Return the indices of only the first few
imgIdx = imgsOfClassIdxs(idx(1:numImgsToShow));

end

function [sortedScores,imgIdx] = findMinActivatingImages(imds,className,predictedScores,numImgsToShow)
% Find the predicted scores of the chosen class on all the images of the chosen class
% (e.g. predicted scores for sushi on all the images of sushi)
[scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores);

% Sort the scores in ascending order
[sortedScores,idx] = sort(scoresForChosenClass,'ascend');

% Return the indices of only the first few
imgIdx = imgsOfClassIdxs(idx(1:numImgsToShow));

end
%%
function [scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores)
% Find the index of className (e.g. "sushi" is the 9th class)
uniqueClasses = unique(imds.Labels);
chosenClassIdx = string(uniqueClasses) == className;

% Find the indices in imageDatastore that are images of label "className"
% (e.g. find all images of class sushi)
imgsOfClassIdxs = find(imds.Labels == className);

% Find the predicted scores of the chosen class on all the images of the
% chosen class
% (e.g. predicted scores for sushi on all the images of sushi)
scoresForChosenClass = predictedScores(imgsOfClassIdxs,chosenClassIdx);
end

function plotImages(imds,imgIdx,sortedScores,predictedClasses,numImgsToShow)

for i=1:numImgsToShow
    score = sortedScores(i);
    sortedImgIdx = imgIdx(i);
    predClass = predictedClasses(sortedImgIdx); 
    correctClass = imds.Labels(sortedImgIdx);
        
    imgPath = imds.Files{sortedImgIdx};
    
    if predClass == correctClass
        color = "\color{green}";
    else
        color = "\color{red}";
    end
    
    predClassTitle = strrep(string(predClass),'_',' ');
    correctClassTitle = strrep(string(correctClass),'_',' ');
    
    subplot(3,ceil(numImgsToShow./3),i)
    imshow(imread(imgPath));
    title("Predicted: " + color + predClassTitle + "\newline\color{black}Score: " + num2str(score) + "\newlineGround truth: " + correctClassTitle);
end

end