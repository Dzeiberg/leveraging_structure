import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook as tqdm
    if 'terminal' in ipy_str:
        from tqdm import tqdm
except:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        def tqdm(iterable, **kwargs):
            return iterable


from LeveragingStructure.experiment_utils import *
from LeveragingStructure.data.leveragingStructure import *
import fire
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from sklearn.metrics import roc_auc_score,silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
import timeit
import shutil

def main(experimentPath,problem_setting=2,labelProportion=.5,minsize=500,
         cluster_range=[1,2,4,8],
         bagLabeledSampleDistribution=lambda n: n,
         bagUnlabeledSampleDistribution=lambda n: n,
         minibatchKMeans=True,
         reassignment_ratio=.001,
         root_dir="/data/dzeiberg/folktable/",
         resampleGroupID=False,
         problem_name="income",
         pca=False,
         allowDuplicates=False,
         kmeans_batch_size=2^12,
         clf="rf",
         nnKwargs={"nModels":5, "NEpochs": 250},
         rfKwargs={"n_estimators":500,"n_jobs":-1, "max_depth": 10},
         tol=.01,
         verbose=True,
         silhouette_bootstrap_size=25000,
         datasetType="acs",
         cache_dir="/data/dzeiberg/huggingface/",
         huggingfaceSaveDir="/data/dzeiberg/leveragingStructureResponseExperiments/",
         bufferAmazon=True,
         syntheticNTargets=100,
         syntheticDim=2,
         syntheticNClusters=2,
         syntheticKwargs=dict(aucRange=[.75,.95],
            irreducibility_range=[.01,.9],
            num_points_labeled_partition=lambda: np.random.normal(1000,100),
            num_points_unlabeled_partition=lambda: np.random.normal(10000,1000),
            timeoutMins=1,
            nTimeouts=2)):
        ##########################
        #       Load Data        #
        ##########################
        if os.path.isdir(experimentPath):
            if os.path.isfile(os.path.join(experimentPath,"fe","preds.npy")):
                print("experiment already exists and appears to have finished, skipping...")
                return
            else:
                print("experiment didn't complete, restarting")
                shutil.move(experimentPath,experimentPath+"_FAILED")
        os.mkdir(experimentPath)
        start_time = timeit.default_timer()
        useXG = clf == "xg"
        useNN = clf == "nn"
        useRF = clf == "rf"
        baseDSKwargs= dict(resampleGroupID=resampleGroupID,
                allowDuplicates=allowDuplicates,
                labelProportion=labelProportion,
                minsize=minsize,
                cluster_range=cluster_range,
                bagLabeledSampleDistribution=bagLabeledSampleDistribution,
                bagUnlabeledSampleDistribution=bagUnlabeledSampleDistribution,
                minibatchKMeans=minibatchKMeans,
                reassignment_ratio=reassignment_ratio,
                batch_size=kmeans_batch_size,
                verbose=verbose,
                tol=tol)

        if problem_name in ["income","employment","income_poverty_ratio"]:
            datasetType = "acs"
            acsKwargs = dict(root_dir=root_dir,problem_name=problem_name)
            if problem_setting == 2:
                d = ACSLoaderSetting2(**baseDSKwargs,**acsKwargs)
            else:
                d = ACSLoaderSetting1(**baseDSKwargs,**acsKwargs)
        elif datasetType == "huggingface":
            if problem_setting == 1:
                d = HuggingfaceDatasetSetting1(pca=pca,cache_dir=cache_dir,huggingfaceSaveDir=huggingfaceSaveDir,bufferAmazon=bufferAmazon,**baseDSKwargs)
            else:
                d = HuggingfaceDatasetSetting2(pca=pca,cache_dir=cache_dir,huggingfaceSaveDir=huggingfaceSaveDir,bufferAmazon=bufferAmazon,**baseDSKwargs)
        elif datasetType == "synthetic":
            if problem_setting == 2:
                d = SyntheticSetting2.from_criteria(syntheticNTargets,syntheticNClusters,syntheticDim,**syntheticKwargs)
            else:
                d = SyntheticSetting1.from_criteria(syntheticNTargets,syntheticNClusters,syntheticDim,**syntheticKwargs)
        else:
            raise ValueError("datasetType",datasetType, "was not implemented")
        XLabeled,yLabeled,instanceNumLabeled,bagLabeled = map(np.concatenate, list(zip(*[s[:]+(s.instanceNum,np.ones(len(s),dtype=int)*sNum) for (sNum,s) in enumerate(d.labeledSamples)])))        
        
        gss = GroupShuffleSplit(n_splits=1)
        
        labeledTrainIndices,labeledValIndices = next(iter(gss.split(XLabeled,yLabeled,instanceNumLabeled)))
        
        XLabeledTrain = XLabeled[labeledTrainIndices]
        XLabeledVal = XLabeled[labeledValIndices]
        yLabeledTrain = yLabeled[labeledTrainIndices]
        yLabeledVal = yLabeled[labeledValIndices]
        instanceNumLabeledTrain = instanceNumLabeled[labeledTrainIndices]
        instanceNumLabeledVal = instanceNumLabeled[labeledValIndices]
        bagLabeledTrain = bagLabeled[labeledTrainIndices]
        bagLabeledVal = bagLabeled[labeledValIndices]
        
        XUnlabeled,yUnlabeled,instanceNumUnlabeled = list(zip(*[s[:]+(s.instanceNum,) for s in d.unlabeledSamples]))
        
        bagUnlabeled = [np.ones(len(y),dtype=int)*bagNum for bagNum,y in enumerate(yUnlabeled)]
        
        XUnlabeled,yUnlabeled,bagUnlabeled,instanceNumUnlabeled = map(np.concatenate, [XUnlabeled,yUnlabeled,bagUnlabeled,instanceNumUnlabeled])
        
        unlabeledTrainIndices,unlabeledTestIndices = next(iter(GroupShuffleSplit(n_splits=1).split(XUnlabeled,
                                                                                                   yUnlabeled,
                                                                                                   instanceNumUnlabeled)))
        
        XUnlabeledTrain,yUnlabeledTrain = XUnlabeled[unlabeledTrainIndices],yUnlabeled[unlabeledTrainIndices]
        bagUnlabeledTrain = bagUnlabeled[unlabeledTrainIndices]
        instanceNumUnlabeledTrain = instanceNumUnlabeled[unlabeledTrainIndices]
        
        XUnlabeledTest,yUnlabeledTest = XUnlabeled[unlabeledTestIndices],yUnlabeled[unlabeledTestIndices]
        bagUnlabeledTest = bagUnlabeled[unlabeledTestIndices]
        instanceNumUnlabeledTest = instanceNumUnlabeled[unlabeledTestIndices]
        
        np.save(os.path.join(experimentPath,"instanceNumLabeledTrain.npy"),instanceNumLabeledTrain)
        np.save(os.path.join(experimentPath,"instanceNumLabeledVal.npy"),instanceNumLabeledVal)
        np.save(os.path.join(experimentPath,"instanceNumUnlabeledTrain.npy"),instanceNumUnlabeledTrain)
        np.save(os.path.join(experimentPath,"instanceNumUnlabeledTest.npy"),instanceNumUnlabeledTest)
        np.save(os.path.join(experimentPath,"bagLabeledTrain.npy"),bagLabeledTrain)
        np.save(os.path.join(experimentPath,"bagLabeledVal.npy"),bagLabeledVal)
        np.save(os.path.join(experimentPath,"bagUnlabeledTrain.npy"),bagUnlabeledTrain)
        np.save(os.path.join(experimentPath,"bagUnlabeledTest.npy"),bagUnlabeledTest)
        np.save(os.path.join(experimentPath,"yUnlabeledTest.npy"),yUnlabeledTest)

        if datasetType == "synthetic":
            for name,f in [("XLabeledTrain",XLabeledTrain),("XLabeledVal",XLabeledVal),
                           ("yLabeledTrain",yLabeledTrain),("yLabeledVal",yLabeledVal),
                           ("XUnlabeledTrain",XUnlabeledTrain),("XUnlabeledTest",XUnlabeledTest),
                           ("yUnlabeledTrain",yUnlabeledTrain)]:
                np.save(os.path.join(experimentPath,f"{name}.npy"),f)

        ###################################
        #  Run Group Aware Global Method    #
        ###################################
        print("Group Aware Global")
        ag = GroupAwareGlobal(os.path.join(experimentPath,"ag"))
        ag.fit(XLabeledTrain,yLabeledTrain,bagLabeledTrain,
               XLabeledVal,yLabeledVal,bagLabeledVal,
               XUnlabeledTrain,bagUnlabeledTrain,
               useNN=useNN,useRF=useRF,useXG=useXG,nnKwargs=nnKwargs,rfKwargs=rfKwargs)
        agPreds = ag.predict(XUnlabeledTest,bagUnlabeledTest)
        np.save(os.path.join(experimentPath,"ag","preds.npy"),agPreds)
        print("~~~~~ Group Aware Global Results ~~~~~~~~~")
        agauc = roc_auc_score(yUnlabeledTest,agPreds)
        print(f"ag: {agauc:.3f}")
        ###################################
        # Run Our Method and its variants #
        ###################################
        print("~~~~~~~~~~~~~~~~~~ Merge Method")
        mm = Method(os.path.join(experimentPath,"mm"))
        mm.fit(XLabeledTrain,yLabeledTrain,XLabeledVal,yLabeledVal,XUnlabeledTrain,bagUnlabeledTrain,
               cluster_range=cluster_range, silhouette_bootstrap_size=silhouette_bootstrap_size, minibatchKMeans=True,
               useXG=useXG, useNN=useNN,useRF=useRF,nnKwargs=nnKwargs,rfKwargs=rfKwargs,
               verbose=verbose,batch_size=kmeans_batch_size,reassignment_ratio=reassignment_ratio,tol=tol)
        print("~~~~~~~~~~~~ MERGE METHOD RESULTS ~~~~~~~~~~~~~~~~~~")
        mmPreds = mm.predict(XUnlabeledTest,bagUnlabeledTest)
        mmClusterGlobalpreds = mm.predict(XUnlabeledTest,bagUnlabeledTest,clusterGlobal=True)
        np.save(os.path.join(experimentPath,"mm","preds.npy"),mmPreds)
        np.save(os.path.join(experimentPath,"mm","clusterGlobalPreds.npy"),mmClusterGlobalpreds)
        for name,auc in zip(["Global", "Our Method"],
                            [roc_auc_score(yUnlabeledTest,mmClusterGlobalpreds),
                             roc_auc_score(yUnlabeledTest,mmPreds)]):
            print(f"{name}: {auc:.3f}\n")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        ###############################################################
        # Run our method with the clustering fit on the original data #
        ###############################################################
        print("~~~~~~~~~~~~~~~~~~ Merge Method With True Clustering")
        mmStar = Method(os.path.join(experimentPath,"mmStar"))
        mmStar.fit(XLabeledTrain,yLabeledTrain,XLabeledVal,yLabeledVal,XUnlabeledTrain,bagUnlabeledTrain,
               cluster_range=cluster_range, silhouette_bootstrap_size=silhouette_bootstrap_size, minibatchKMeans=True,
               useXG=useXG, useNN=useNN,useRF=useRF,nnKwargs=nnKwargs,rfKwargs=rfKwargs,
               verbose=verbose,batch_size=kmeans_batch_size,reassignment_ratio=reassignment_ratio,tol=tol,
               estimatedClustering=d.trueClusterer)
        print("~~~~~~~~~~~~ MERGE METHOD With True Clustering RESULTS ~~~~~~~~~~~~~~~~~~")
        mmStarClusterGlobalpreds = mmStar.predict(XUnlabeledTest,bagUnlabeledTest,clusterGlobal=True)
        mmStarPreds = mmStar.predict(XUnlabeledTest,bagUnlabeledTest)
        np.save(os.path.join(experimentPath,"mmStar","clusterGlobalPreds.npy"),mmStarClusterGlobalpreds)
        np.save(os.path.join(experimentPath,"mmStar","preds.npy"),mmStarPreds)
        for name,auc in zip(["Global", "Our Method"],
                            [roc_auc_score(yUnlabeledTest,mmStarClusterGlobalpreds),
                             roc_auc_score(yUnlabeledTest,mmStarPreds)]):
            print(f"{name}: {auc:.3f}\n")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        ###########################
        # Single CLuster Ablation #
        ###########################
        print("~~~~~~~~~~~~~~~~~~~ K1 Ablation")
        mm2 = Method(os.path.join(experimentPath,"mm2"))
        mm2.fit(XLabeledTrain,yLabeledTrain,XLabeledVal,yLabeledVal,XUnlabeledTrain,bagUnlabeledTrain,
                cluster_range=[1],silhouette_bootstrap_size=silhouette_bootstrap_size, minibatchKMeans=True,
                useXG=useXG, useNN=useNN,useRF=useRF,nnKwargs=nnKwargs,rfKwargs=rfKwargs,
                verbose=verbose,batch_size=kmeans_batch_size,reassignment_ratio=reassignment_ratio,tol=tol)
        mm2GlobalPreds = mm2.predict(XUnlabeledTest,bagUnlabeledTest,clusterGlobal=True)
        mm2Preds = mm2.predict(XUnlabeledTest,bagUnlabeledTest)
        np.save(os.path.join(experimentPath,"mm2","clusterGlobalPreds.npy"),mm2GlobalPreds)
        np.save(os.path.join(experimentPath,"mm2","Preds.npy"),mm2Preds)
        print("~~~~~~~~~~~~~ K1 Ablation RESULTS ~~~~~~~~~~~~~~~~~~~~")
        for name,auc in zip(["Global", "Our Method"],[roc_auc_score(yUnlabeledTest,mm2GlobalPreds),
            roc_auc_score(yUnlabeledTest,mm2Preds)]):
            print(f"{name}: {auc:.3f}\n")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        ###########################
        # Frustratingly Easy Domain Adaptation #
        ###########################
        print("~~~~~~~~~~~~~~~~~~~ Frustratingly Easy")
        fe = FrustratinglyEasyDomainAdaptation(os.path.join(experimentPath,"fe"))
        fe.fit(XLabeledTrain,yLabeledTrain,XLabeledVal,yLabeledVal,XUnlabeledTrain,bagUnlabeledTrain,rfKwargs=rfKwargs)
        fePreds = fe.predict(XUnlabeledTest,bagUnlabeledTest)
        np.save(os.path.join(experimentPath,"fe","preds.npy"),fePreds)
        stop_time = timeit.default_timer()
        print(f"Runtime {stop_time - start_time} sec.")

if __name__ == '__main__':
    fire.Fire(main)
