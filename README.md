# Leveraging Structure for Improved Classification of Grouped Data



## Create environment and Install Dependencies

`conda create --name structure python=3.9`

`conda activate structure`

`python -m pip install -r requirements.txt`

# Running Experiment on Synthetic Data

## Create Synthetic Dataset

```python
from LeveragingStructure.data.leveragingStructure import SyntheticSetting2
```

```python
import numpy as np
```

```python
dataset = SyntheticSetting2.from_criteria(n_targets=10,n_clusters=2,dim=2,aucRange=[.75,.95],
                                          irreducibility_range=[.01,.9],
                                         num_points_labeled_partition=lambda: np.round(np.random.normal(1000,100)),
                                         num_points_unlabeled_partition=lambda: np.round(np.random.normal(10000,1000)),
                                         timeoutMins=2,nTimeouts=3)
```

    generating components
    making dg


    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 58.69it/s]

    isMetricSatisfied{'aucpn': 0.5, 'irreducibility': [0.0, 0.0]}
    anchorSetProp0.01
    not satisfied
    current time: 1660849235.7673945 - will time out at 1660849355.7611418
    Mu Perturb
    isMetricSatisfied{'aucpn': 0.76098348, 'irreducibility': [0.046, 0.0462]}
    anchorSetProp0.01
    isMetricSatisfied{'aucpn': 0.5, 'irreducibility': [0.0, 0.0]}
    anchorSetProp0.01
    not satisfied
    current time: 1660849235.783332 - will time out at 1660849355.7789536
    Mu Perturb
    isMetricSatisfied{'aucpn': 0.75581508, 'irreducibility': [0.0478, 0.0406]}
    anchorSetProp0.01
    done making dg
    jittering comps
    getting conflicting pairs
    conflicting pairs
    done getting conflicting pairs
    [(0, 1)]
    [[ 1.2681453  -1.56331616]
     [ 1.30115567  0.19468389]]


    


    [[-0.18615782 -2.56357543]
     [ 1.87638927  1.79496946]]
    done jittering comps
    done generating components


      0%|                                                                                                                                                         | 0/10 [00:00<?, ?it/s]
      0%|                                                                                                                                                          | 0/2 [00:00<?, ?it/s][A
                                                                                                                                                                                         [A
      0%|                                                                                                                                                          | 0/2 [00:00<?, ?it/s][A
                                                                                                                                                                                         [A
      0%|                                                                                                                                                          | 0/2 [00:00<?, ?it/s][A
                                                                                                                                                                                         [A
      0%|                                                                                                                                                          | 0/2 [00:00<?, ?it/s][A
                                                                                                                                                                                         [A
      0%|                                                                                                                                                          | 0/2 [00:00<?, ?it/s][A
                                                                                                                                                                                         [A
      0%|                                                                                                                                                          | 0/2 [00:00<?, ?it/s][A
                                                                                                                                                                                         [A
      0%|                                                                                                                                                          | 0/2 [00:00<?, ?it/s][A
                                                                                                                                                                                         [A
      0%|                                                                                                                                                          | 0/2 [00:00<?, ?it/s][A
                                                                                                                                                                                         [A
      0%|                                                                                                                                                          | 0/2 [00:00<?, ?it/s][A
                                                                                                                                                                                         [A
      0%|                                                                                                                                                          | 0/2 [00:00<?, ?it/s][A
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 121.34it/s][A

    done from components


    


### Visualize (Synthetic Dataset Only)

```python
dataset.NMix.plotCIEllipse()
```


    
![png](docs/images/output_10_0.png)
    


## Split Labeled and Unlabeled Sets

```python
from sklearn.model_selection import GroupShuffleSplit
```

### Split Labeled Train and Validation Sets

```python
XLabeled,yLabeled,instanceNumLabeled,bagLabeled = map(np.concatenate, list(zip(*[s[:]+(s.instanceNum,
                                                                                       np.ones(len(s),
                                                                                               dtype=int)*sNum) \
                                                                                 for (sNum,s) in enumerate(dataset.labeledSamples)])))
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
```

### Split Unlabeled Train and Test Sets

```python
XUnlabeled,yUnlabeled,instanceNumUnlabeled = list(zip(*[s[:]+(s.instanceNum,) for s in dataset.unlabeledSamples]))

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
```

## Run Our Method

```python
import os
if not os.path.isdir("experiments"):
    os.mkdir("experiments")
```

```python
from LeveragingStructure.experiment_utils import Method
```

    2022-08-18 15:00:42.065562: I tensorflow/core/util/util.cc:169] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2022-08-18 15:00:42.070297: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda-10.1/lib64
    2022-08-18 15:00:42.070315: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.


```python
method = Method("experiments/synthetic_experiment")
```

```python
method.fit(XLabeledTrain,yLabeledTrain,XLabeledVal,yLabeledVal,XUnlabeledTrain,bagUnlabeledTrain,
           cluster_range=np.arange(1,4))
```

      0%|                                                                                                                                                          | 0/3 [00:00<?, ?it/s]

    Init 1/3 with method k-means++
    Inertia for init 1/3: 264.26728982698444
    Init 2/3 with method k-means++
    Inertia for init 2/3: 143.43994473159813
    Init 3/3 with method k-means++
    Inertia for init 3/3: 98.15466763773992
    Minibatch step 1/674000: mean batch inertia: 3.746532991366581
    Minibatch step 2/674000: mean batch inertia: 2.6886901910044587, ewa inertia: 2.6886901910044587
    Minibatch step 3/674000: mean batch inertia: 2.076976120648504, ewa inertia: 2.688508675399813
    Converged (small centers change) at step 3/674000


     67%|█████████████████████████████████████████████████████████████████████████████████████████████████▎                                                | 2/3 [00:00<00:00,  3.25it/s]

    Init 1/3 with method k-means++
    Inertia for init 1/3: 111.73438433875798
    Init 2/3 with method k-means++
    Inertia for init 2/3: 92.98178570026872
    Init 3/3 with method k-means++
    Inertia for init 3/3: 81.36216234753705
    Minibatch step 1/674000: mean batch inertia: 2.337100796670699
    Minibatch step 2/674000: mean batch inertia: 2.263008775829778, ewa inertia: 2.263008775829778
    Minibatch step 3/674000: mean batch inertia: 2.050856384911293, ewa inertia: 2.262945823265204
    Minibatch step 4/674000: mean batch inertia: 1.6897037431290423, ewa inertia: 2.262775723560413
    Minibatch step 5/674000: mean batch inertia: 1.918077105333506, ewa inertia: 2.2626734401879354
    Converged (small centers change) at step 5/674000


    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.46it/s]


    [0, 0.6345693339685641, 0.44819267917302874]
    found 2 clusters...
    Init 1/3 with method k-means++
    Inertia for init 1/3: 221.57178149186015
    Init 2/3 with method k-means++
    Inertia for init 2/3: 130.53784753623984
    Init 3/3 with method k-means++
    Inertia for init 3/3: 152.8192145547951
    Minibatch step 1/674000: mean batch inertia: 3.3106371493084366
    Minibatch step 2/674000: mean batch inertia: 2.956770115134291, ewa inertia: 2.956770115134291
    Minibatch step 3/674000: mean batch inertia: 2.8485671633838856, ewa inertia: 2.9567380077737395
    Minibatch step 4/674000: mean batch inertia: 2.9704351082672287, ewa inertia: 2.9567420721521778
    Minibatch step 5/674000: mean batch inertia: 2.5673096887142783, ewa inertia: 2.9566265148060684
    Minibatch step 6/674000: mean batch inertia: 2.159528471207144, ewa inertia: 2.956389989703317
    Minibatch step 7/674000: mean batch inertia: 1.5922525747247034, ewa inertia: 2.9559852054426647
    Converged (small centers change) at step 7/674000


     50%|█████████████████████████████████████████████████████████████████████████                                                                         | 1/2 [00:00<00:00,  1.14it/s]

    Cluster 0 validation auc: 0.729


    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.04it/s]

    Cluster 1 validation auc: 0.742


    
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  4.44it/s]


## Evaluate Method

```python
from sklearn.metrics import roc_auc_score
```

```python
roc_auc_score(yUnlabeledTest,method.predict(XUnlabeledTest,bagUnlabeledTest))
```




    0.850584575506158



# Experiment on ACS Data

```python
from LeveragingStructure.data.leveragingStructure import ACSLoaderSetting2
```

```python
baseDSKwargs= dict(resampleGroupID=False,
                allowDuplicates=False,
                labelProportion=.5,
                minsize=500,
                cluster_range=np.arange(1,8),
                bagLabeledSampleDistribution=lambda bag_size: bag_size,
                bagUnlabeledSampleDistribution=lambda bag_size: bag_size,
                minibatchKMeans=True,
                reassignment_ratio=.001,
                batch_size=2^13,
                verbose=True,
                tol=.01)
dataset2 = ACSLoaderSetting2(**baseDSKwargs, problem_name="income",root_dir="./acsrootdir/")
```

    /home/dzeiberg/MultiInstancePU/LeveragingStructure/data/leveragingStructure.py:304: UserWarning: making structured pool
      warn("making structured pool")
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:14<00:00,  3.45it/s]


    mapping states to problem


    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [01:38<00:00,  1.93s/it]


    creating matrices
    8116931 total instances


    /home/dzeiberg/MultiInstancePU/LeveragingStructure/data/leveragingStructure.py:310: UserWarning: partitioning dataset
      warn("partitioning dataset")
    /home/dzeiberg/MultiInstancePU/LeveragingStructure/data/leveragingStructure.py:312: UserWarning: splitting partitions
      warn("splitting partitions")
    /home/dzeiberg/MultiInstancePU/LeveragingStructure/data/leveragingStructure.py:314: UserWarning: resampling partitions
      warn("resampling partitions")
      0%|                                                                                                                                                          | 0/7 [00:00<?, ?it/s]

    using minibatch
    Init 1/3 with method k-means++
    Inertia for init 1/3: 826.9317916677008
    Init 2/3 with method k-means++
    Inertia for init 2/3: 625.9105004281943
    Init 3/3 with method k-means++
    Inertia for init 3/3: 643.0486087981654
    Minibatch step 1/54112873: mean batch inertia: 16.013699057056566
    Minibatch step 2/54112873: mean batch inertia: 8.369301145354745, ewa inertia: 8.369301145354745
    Minibatch step 3/54112873: mean batch inertia: 6.301646188321194, ewa inertia: 8.369293503348048
    Minibatch step 4/54112873: mean batch inertia: 7.324371817046947, ewa inertia: 8.369289641340755
    Minibatch step 5/54112873: mean batch inertia: 10.001702373791385, ewa inertia: 8.36929567470188
    Minibatch step 6/54112873: mean batch inertia: 9.973441751685126, ewa inertia: 8.369301603590074
    Minibatch step 7/54112873: mean batch inertia: 9.140024390498679, ewa inertia: 8.36930445216434
    Minibatch step 8/54112873: mean batch inertia: 7.702952790784951, ewa inertia: 8.36930198934343
    Minibatch step 9/54112873: mean batch inertia: 8.49865015989304, ewa inertia: 8.36930246741139
    Converged (small centers change) at step 9/54112873
    predicting...


     29%|█████████████████████████████████████████▋                                                                                                        | 2/7 [00:01<00:04,  1.04it/s]

    using minibatch
    Init 1/3 with method k-means++
    Inertia for init 1/3: 604.6139022570061
    Init 2/3 with method k-means++
    Inertia for init 2/3: 532.5620400798554
    Init 3/3 with method k-means++
    Inertia for init 3/3: 564.7436972058847
    Minibatch step 1/54112873: mean batch inertia: 13.955880526446816
    Minibatch step 2/54112873: mean batch inertia: 11.362955274013496, ewa inertia: 11.362955274013496
    Minibatch step 3/54112873: mean batch inertia: 9.314257546238803, ewa inertia: 11.362947702072296
    Minibatch step 4/54112873: mean batch inertia: 7.272690138299981, ewa inertia: 11.362932584571384
    Minibatch step 5/54112873: mean batch inertia: 10.829263581692953, ewa inertia: 11.362930612142629
    Minibatch step 6/54112873: mean batch inertia: 9.448255516472718, ewa inertia: 11.362923535546093
    Minibatch step 7/54112873: mean batch inertia: 8.650492753062673, ewa inertia: 11.362913510462297
    Minibatch step 8/54112873: mean batch inertia: 9.78757545967264, ewa inertia: 11.362907688047928
    Minibatch step 9/54112873: mean batch inertia: 7.811010003194126, ewa inertia: 11.362894560313146
    Minibatch step 10/54112873: mean batch inertia: 8.69575148048161, ewa inertia: 11.36288470261169
    Minibatch step 11/54112873: mean batch inertia: 9.556314870736138, ewa inertia: 11.362878025569804
    Minibatch step 12/54112873: mean batch inertia: 8.058137924202088, ewa inertia: 11.36286581132395
    Minibatch step 13/54112873: mean batch inertia: 6.3351071649052475, ewa inertia: 11.362847228839904
    Minibatch step 14/54112873: mean batch inertia: 7.383677537457887, ewa inertia: 11.362832521917294
    Minibatch step 15/54112873: mean batch inertia: 8.822437906218482, ewa inertia: 11.36282313267534
    Minibatch step 16/54112873: mean batch inertia: 7.684396026874716, ewa inertia: 11.362809537290634
    Minibatch step 17/54112873: mean batch inertia: 6.235601412047429, ewa inertia: 11.362790587243527
    Minibatch step 18/54112873: mean batch inertia: 8.14588743850671, ewa inertia: 11.362778697641094
    Minibatch step 19/54112873: mean batch inertia: 10.01362026698498, ewa inertia: 11.362773711181566
    Minibatch step 20/54112873: mean batch inertia: 6.509428245814075, ewa inertia: 11.362755773324755
    Minibatch step 21/54112873: mean batch inertia: 7.191553596281912, ewa inertia: 11.362740356654355
    Converged (small centers change) at step 21/54112873
    predicting...


     43%|██████████████████████████████████████████████████████████████▌                                                                                   | 3/7 [00:04<00:06,  1.57s/it]

    using minibatch
    Init 1/3 with method k-means++
    Inertia for init 1/3: 443.63998646400614
    Init 2/3 with method k-means++
    Inertia for init 2/3: 536.5008590030448
    Init 3/3 with method k-means++
    Inertia for init 3/3: 480.25401600418365
    Minibatch step 1/54112873: mean batch inertia: 11.548454501520759
    Minibatch step 2/54112873: mean batch inertia: 7.220564132062753, ewa inertia: 7.220564132062753
    Minibatch step 3/54112873: mean batch inertia: 6.953503017404995, ewa inertia: 7.220563145010818
    Minibatch step 4/54112873: mean batch inertia: 5.146476364341405, ewa inertia: 7.220555479232243
    Minibatch step 5/54112873: mean batch inertia: 8.304356832679073, ewa inertia: 7.220559484937922
    Minibatch step 6/54112873: mean batch inertia: 11.894134870270594, ewa inertia: 7.220576758368519
    Minibatch step 7/54112873: mean batch inertia: 9.880342835429468, ewa inertia: 7.220586588804738
    Minibatch step 8/54112873: mean batch inertia: 7.705218529616652, ewa inertia: 7.220588379993604
    Minibatch step 9/54112873: mean batch inertia: 7.913184842332539, ewa inertia: 7.220590939814713
    Minibatch step 10/54112873: mean batch inertia: 7.129624068936995, ewa inertia: 7.220590603603183
    Minibatch step 11/54112873: mean batch inertia: 7.163055961945232, ewa inertia: 7.220590390956428
    Minibatch step 12/54112873: mean batch inertia: 9.184423221219856, ewa inertia: 7.220597649238856
    Minibatch step 13/54112873: mean batch inertia: 6.837257764881966, ewa inertia: 7.220596232423176
    Minibatch step 14/54112873: mean batch inertia: 6.742561142753956, ewa inertia: 7.220594465616125
    Converged (lack of improvement in inertia) at step 14/54112873
    predicting...


     57%|███████████████████████████████████████████████████████████████████████████████████▍                                                              | 4/7 [00:06<00:05,  1.75s/it]

    using minibatch
    Init 1/3 with method k-means++
    Inertia for init 1/3: 520.0889359463417
    Init 2/3 with method k-means++
    Inertia for init 2/3: 395.07831785449736
    Init 3/3 with method k-means++
    Inertia for init 3/3: 410.4900484817321
    Minibatch step 1/54112873: mean batch inertia: 14.217362314797871
    Minibatch step 2/54112873: mean batch inertia: 8.062423848381004, ewa inertia: 8.062423848381004
    Minibatch step 3/54112873: mean batch inertia: 9.57061937754251, ewa inertia: 8.06242942263811
    Minibatch step 4/54112873: mean batch inertia: 6.877504458917745, ewa inertia: 8.062425043181818
    Minibatch step 5/54112873: mean batch inertia: 8.329140871038451, ewa inertia: 8.062426028957582
    Minibatch step 6/54112873: mean batch inertia: 9.249522936935538, ewa inertia: 8.06243041644133
    Minibatch step 7/54112873: mean batch inertia: 10.19567113569144, ewa inertia: 8.062438300851545
    Minibatch step 8/54112873: mean batch inertia: 8.701055060379662, ewa inertia: 8.062440661164874
    Minibatch step 9/54112873: mean batch inertia: 8.119840553549052, ewa inertia: 8.0624408733136
    Minibatch step 10/54112873: mean batch inertia: 5.923851394716282, ewa inertia: 8.06243296913449
    Minibatch step 11/54112873: mean batch inertia: 8.920838775355874, ewa inertia: 8.062436141783243
    Minibatch step 12/54112873: mean batch inertia: 7.89502561232331, ewa inertia: 8.062435523037653
    Converged (lack of improvement in inertia) at step 12/54112873
    predicting...


     71%|████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                         | 5/7 [00:08<00:03,  1.93s/it]

    using minibatch
    Init 1/3 with method k-means++
    Inertia for init 1/3: 454.5920143685284
    Init 2/3 with method k-means++
    Inertia for init 2/3: 455.3313243959065
    Init 3/3 with method k-means++
    Inertia for init 3/3: 385.200344791383
    Minibatch step 1/54112873: mean batch inertia: 10.75144695586156
    Minibatch step 2/54112873: mean batch inertia: 10.057375264795125, ewa inertia: 10.057375264795125
    Minibatch step 3/54112873: mean batch inertia: 8.67187110693317, ewa inertia: 10.057370144002597
    Minibatch step 4/54112873: mean batch inertia: 8.710352917533394, ewa inertia: 10.057365165456911
    Minibatch step 5/54112873: mean batch inertia: 5.469047415473325, ewa inertia: 10.057348207136638
    Minibatch step 6/54112873: mean batch inertia: 6.234742697435007, ewa inertia: 10.05733407887176
    Minibatch step 7/54112873: mean batch inertia: 7.78184205264802, ewa inertia: 10.05732566870388
    Minibatch step 8/54112873: mean batch inertia: 6.766361601958861, ewa inertia: 10.057313505373942
    Minibatch step 9/54112873: mean batch inertia: 11.176276040448414, ewa inertia: 10.057317641034565
    Minibatch step 10/54112873: mean batch inertia: 6.429544960324036, ewa inertia: 10.057304232867487
    Minibatch step 11/54112873: mean batch inertia: 6.509786762898997, ewa inertia: 10.057291121321882
    Minibatch step 12/54112873: mean batch inertia: 10.393094741161018, ewa inertia: 10.057292362444587
    Minibatch step 13/54112873: mean batch inertia: 7.784173764246066, ewa inertia: 10.057283961048844
    Minibatch step 14/54112873: mean batch inertia: 5.522606845504058, ewa inertia: 10.057267200983162
    Minibatch step 15/54112873: mean batch inertia: 6.751787186077106, ewa inertia: 10.057254984002604
    Minibatch step 16/54112873: mean batch inertia: 6.0006950034033855, ewa inertia: 10.05723999104721
    Minibatch step 17/54112873: mean batch inertia: 5.20635617148545, ewa inertia: 10.057222062288588
    Minibatch step 18/54112873: mean batch inertia: 5.683154212655512, ewa inertia: 10.057205895831178
    Minibatch step 19/54112873: mean batch inertia: 5.8834461952682675, ewa inertia: 10.057190469708226
    Minibatch step 20/54112873: mean batch inertia: 6.9636911722245065, ewa inertia: 10.057179036203681
    Minibatch step 21/54112873: mean batch inertia: 6.7426657179553375, ewa inertia: 10.057166785836234
    Minibatch step 22/54112873: mean batch inertia: 7.698493870683936, ewa inertia: 10.05715806823364
    Minibatch step 23/54112873: mean batch inertia: 6.963359271907658, ewa inertia: 10.057146633622153
    Minibatch step 24/54112873: mean batch inertia: 5.694676206199899, ewa inertia: 10.057130510028557
    Minibatch step 25/54112873: mean batch inertia: 5.207771492786495, ewa inertia: 10.05711258690557
    Minibatch step 26/54112873: mean batch inertia: 6.514530462261746, ewa inertia: 10.05709949360089
    Minibatch step 27/54112873: mean batch inertia: 6.18665699526696, ewa inertia: 10.057085188531568
    Minibatch step 28/54112873: mean batch inertia: 7.260381139076805, ewa inertia: 10.057074851975653
    Minibatch step 29/54112873: mean batch inertia: 7.2090703724084655, ewa inertia: 10.05706432581449
    Minibatch step 30/54112873: mean batch inertia: 7.3654728314831175, ewa inertia: 10.05705437775224
    Minibatch step 31/54112873: mean batch inertia: 5.499713272183098, ewa inertia: 10.057037533920953
    Minibatch step 32/54112873: mean batch inertia: 5.436777671662815, ewa inertia: 10.057020457543343
    Minibatch step 33/54112873: mean batch inertia: 7.68426593656399, ewa inertia: 10.05701168789545
    Minibatch step 34/54112873: mean batch inertia: 5.67985327841363, ewa inertia: 10.0569955100154
    Minibatch step 35/54112873: mean batch inertia: 6.139833169184251, ewa inertia: 10.05698103227058
    Minibatch step 36/54112873: mean batch inertia: 4.122351832428198, ewa inertia: 10.056959098013154
    Minibatch step 37/54112873: mean batch inertia: 6.63729700429222, ewa inertia: 10.056946459018173
    Minibatch step 38/54112873: mean batch inertia: 7.982287811304423, ewa inertia: 10.05693879112599
    Minibatch step 39/54112873: mean batch inertia: 6.774826815616089, ewa inertia: 10.056926660513184
    Minibatch step 40/54112873: mean batch inertia: 5.1928650187038725, ewa inertia: 10.056908683049624
    Minibatch step 41/54112873: mean batch inertia: 10.23364873225012, ewa inertia: 10.056909336276911
    Minibatch step 42/54112873: mean batch inertia: 6.846580004162302, ewa inertia: 10.056897470971158
    Minibatch step 43/54112873: mean batch inertia: 5.698722672604771, ewa inertia: 10.056881363254112
    Converged (small centers change) at step 43/54112873
    predicting...


     86%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                    | 6/7 [00:11<00:02,  2.31s/it]

    using minibatch
    Init 1/3 with method k-means++
    Inertia for init 1/3: 396.8488375142457
    Init 2/3 with method k-means++
    Inertia for init 2/3: 368.9798458033028
    Init 3/3 with method k-means++
    Inertia for init 3/3: 408.6934684235456
    Minibatch step 1/54112873: mean batch inertia: 8.118617620499993
    Minibatch step 2/54112873: mean batch inertia: 8.105289124819324, ewa inertia: 8.105289124819324
    Minibatch step 3/54112873: mean batch inertia: 6.561874035354469, ewa inertia: 8.105283420391508
    Minibatch step 4/54112873: mean batch inertia: 8.915230405865781, ewa inertia: 8.10528641393754
    Minibatch step 5/54112873: mean batch inertia: 6.213690891195624, ewa inertia: 8.105279422642592
    Minibatch step 6/54112873: mean batch inertia: 6.5182128383321025, ewa inertia: 8.105273556879823
    Minibatch step 7/54112873: mean batch inertia: 7.495871174235295, ewa inertia: 8.105271304542182
    Minibatch step 8/54112873: mean batch inertia: 7.5290207669408264, ewa inertia: 8.10526917473302
    Minibatch step 9/54112873: mean batch inertia: 6.412836117475009, ewa inertia: 8.105262919538113
    Minibatch step 10/54112873: mean batch inertia: 4.8776693635418455, ewa inertia: 8.105250990424173
    Minibatch step 11/54112873: mean batch inertia: 6.078374508549161, ewa inertia: 8.105243499133811
    Minibatch step 12/54112873: mean batch inertia: 6.458038476790729, ewa inertia: 8.105237411100715
    Minibatch step 13/54112873: mean batch inertia: 4.028039763439101, ewa inertia: 8.105222341868963
    Minibatch step 14/54112873: mean batch inertia: 5.043759253063475, ewa inertia: 8.105211026769531
    Minibatch step 15/54112873: mean batch inertia: 5.300170670185995, ewa inertia: 8.10520065940281
    Minibatch step 16/54112873: mean batch inertia: 7.297873462527371, ewa inertia: 8.10519767553946
    Minibatch step 17/54112873: mean batch inertia: 7.284434795964111, ewa inertia: 8.105194642018127
    Minibatch step 18/54112873: mean batch inertia: 6.712477254726147, ewa inertia: 8.105189494565662
    Minibatch step 19/54112873: mean batch inertia: 8.360079270902697, ewa inertia: 8.105190436632602
    Minibatch step 20/54112873: mean batch inertia: 6.34795267332835, ewa inertia: 8.105183941920943
    Minibatch step 21/54112873: mean batch inertia: 6.298639899168513, ewa inertia: 8.105177264974373
    Minibatch step 22/54112873: mean batch inertia: 5.511424393256462, ewa inertia: 8.105167678521493
    Minibatch step 23/54112873: mean batch inertia: 6.135183927890414, ewa inertia: 8.1051603975054
    Minibatch step 24/54112873: mean batch inertia: 5.589444505528113, ewa inertia: 8.10515109947546
    Minibatch step 25/54112873: mean batch inertia: 5.945445293475257, ewa inertia: 8.105143117250874
    Minibatch step 26/54112873: mean batch inertia: 7.168743036817635, ewa inertia: 8.10513965633702
    Minibatch step 27/54112873: mean batch inertia: 4.963397403685809, ewa inertia: 8.105128044527584
    Minibatch step 28/54112873: mean batch inertia: 11.272869078944764, ewa inertia: 8.105139752427938
    Minibatch step 29/54112873: mean batch inertia: 8.526132243897901, ewa inertia: 8.105141308406816
    Minibatch step 30/54112873: mean batch inertia: 6.139172655854103, ewa inertia: 8.105134042230437
    Minibatch step 31/54112873: mean batch inertia: 8.623498304701618, ewa inertia: 8.105135958093213
    Minibatch step 32/54112873: mean batch inertia: 5.471302434402812, ewa inertia: 8.105126223503134
    Minibatch step 33/54112873: mean batch inertia: 8.65507069453656, ewa inertia: 8.105128256085658
    Minibatch step 34/54112873: mean batch inertia: 6.428045876314134, ewa inertia: 8.105122057626511
    Minibatch step 35/54112873: mean batch inertia: 6.166800853074117, ewa inertia: 8.105114893634484
    Minibatch step 36/54112873: mean batch inertia: 5.3848516180818695, ewa inertia: 8.105104839601967
    Minibatch step 37/54112873: mean batch inertia: 7.545262873314813, ewa inertia: 8.105102770438522
    Minibatch step 38/54112873: mean batch inertia: 6.092763654618711, ewa inertia: 8.105095332877942
    Minibatch step 39/54112873: mean batch inertia: 4.7951820512030885, ewa inertia: 8.105083099512129
    Converged (small centers change) at step 39/54112873
    predicting...


    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:14<00:00,  2.08s/it]


    silhouette scores [1 2 3 4 5 6 7] 
     [0.         0.33370916 0.14554513 0.1200664  0.10554993 0.17319521
     0.14454597]
    using minibatch for final clustering
    Init 1/3 with method k-means++
    Inertia for init 1/3: 629.7774664160212
    Init 2/3 with method k-means++
    Inertia for init 2/3: 622.9944032239844
    Init 3/3 with method k-means++
    Inertia for init 3/3: 869.966697396862
    Minibatch step 1/54112873: mean batch inertia: 15.78601915805319
    Minibatch step 2/54112873: mean batch inertia: 11.304651573135454, ewa inertia: 11.304651573135454
    Minibatch step 3/54112873: mean batch inertia: 11.495051398582955, ewa inertia: 11.30465227684897
    Minibatch step 4/54112873: mean batch inertia: 11.758478769936918, ewa inertia: 11.304653954181587
    Minibatch step 5/54112873: mean batch inertia: 7.704030280192082, ewa inertia: 11.304640646356633
    Minibatch step 6/54112873: mean batch inertia: 11.872478907998435, ewa inertia: 11.30464274507421
    Minibatch step 7/54112873: mean batch inertia: 8.832202392839593, ewa inertia: 11.304633606989702
    Minibatch step 8/54112873: mean batch inertia: 9.27566714412196, ewa inertia: 11.30462610797482
    Minibatch step 9/54112873: mean batch inertia: 8.311745172668015, ewa inertia: 11.3046150463535
    Minibatch step 10/54112873: mean batch inertia: 9.357571673484891, ewa inertia: 11.304607850124533
    Minibatch step 11/54112873: mean batch inertia: 9.700755300509236, ewa inertia: 11.304601922321211
    Minibatch step 12/54112873: mean batch inertia: 9.188700753464015, ewa inertia: 11.304594101997589
    Minibatch step 13/54112873: mean batch inertia: 8.885985523953593, ewa inertia: 11.30458516287412
    Minibatch step 14/54112873: mean batch inertia: 10.030477733107727, ewa inertia: 11.304580453801421
    Minibatch step 15/54112873: mean batch inertia: 8.94320889918631, ewa inertia: 11.304571726224717
    Minibatch step 16/54112873: mean batch inertia: 6.459962862017798, ewa inertia: 11.30455382065819
    Converged (small centers change) at step 16/54112873
    Found 2 clusters


    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:11<00:00,  4.42it/s]


```python
XLabeled,yLabeled,instanceNumLabeled,bagLabeled = map(np.concatenate, list(zip(*[s[:]+(s.instanceNum,
                                                                                       np.ones(len(s),
                                                                                               dtype=int)*sNum) \
                                                                                 for (sNum,s) in enumerate(dataset.labeledSamples)])))
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
```

```python
XUnlabeled,yUnlabeled,instanceNumUnlabeled = list(zip(*[s[:]+(s.instanceNum,) for s in dataset.unlabeledSamples]))

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
```

```python
method2 = Method("experiments/acs_income_experiment")
```

```python
method2.fit(XLabeledTrain,yLabeledTrain,XLabeledVal,yLabeledVal,XUnlabeledTrain,bagUnlabeledTrain,cluster_range=np.arange(1,5))
```

      0%|                                                                                                                                                          | 0/4 [00:00<?, ?it/s]

    Init 1/3 with method k-means++
    Inertia for init 1/3: 124.77464253274512
    Init 2/3 with method k-means++
    Inertia for init 2/3: 122.83319903457891
    Init 3/3 with method k-means++
    Inertia for init 3/3: 105.07517055342271
    Minibatch step 1/674000: mean batch inertia: 2.6479343675171165
    Minibatch step 2/674000: mean batch inertia: 3.8244440087049703, ewa inertia: 3.8244440087049703
    Minibatch step 3/674000: mean batch inertia: 1.9110442246317796, ewa inertia: 3.8238762403053768
    Converged (small centers change) at step 3/674000


     50%|█████████████████████████████████████████████████████████████████████████                                                                         | 2/4 [00:00<00:00,  3.28it/s]

    Init 1/3 with method k-means++
    Inertia for init 1/3: 81.25192561391391
    Init 2/3 with method k-means++
    Inertia for init 2/3: 77.28801041985662
    Init 3/3 with method k-means++
    Inertia for init 3/3: 112.82340638203097
    Minibatch step 1/674000: mean batch inertia: 1.869380296249273
    Minibatch step 2/674000: mean batch inertia: 2.6829385080985158, ewa inertia: 2.6829385080985158
    Minibatch step 3/674000: mean batch inertia: 2.345698124418781, ewa inertia: 2.682838437828563
    Converged (small centers change) at step 3/674000


     75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                    | 3/4 [00:01<00:00,  2.35it/s]

    Init 1/3 with method k-means++
    Inertia for init 1/3: 59.44185758566471
    Init 2/3 with method k-means++
    Inertia for init 2/3: 89.11902560999854
    Init 3/3 with method k-means++
    Inertia for init 3/3: 64.36034514631405
    Minibatch step 1/674000: mean batch inertia: 1.2822026115895901
    Minibatch step 2/674000: mean batch inertia: 1.9689995509096632, ewa inertia: 1.9689995509096632
    Minibatch step 3/674000: mean batch inertia: 1.5761644199233622, ewa inertia: 1.9688829838568807
    Minibatch step 4/674000: mean batch inertia: 1.541715639419072, ewa inertia: 1.9687562293116314
    Minibatch step 5/674000: mean batch inertia: 1.1435731889419956, ewa inertia: 1.9685113704702633
    Converged (small centers change) at step 5/674000


    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.19it/s]


    [0, 0.6280206870992453, 0.4671332301754058, 0.31853254981771456]
    found 2 clusters...
    Init 1/3 with method k-means++
    Inertia for init 1/3: 152.6630926411599
    Init 2/3 with method k-means++
    Inertia for init 2/3: 238.18972989984198
    Init 3/3 with method k-means++
    Inertia for init 3/3: 163.33446039311255
    Minibatch step 1/674000: mean batch inertia: 2.684004741181202
    Minibatch step 2/674000: mean batch inertia: 2.261001860695597, ewa inertia: 2.261001860695597
    Minibatch step 3/674000: mean batch inertia: 3.564497218410371, ewa inertia: 2.2613886504712037
    Minibatch step 4/674000: mean batch inertia: 2.2888688254821816, ewa inertia: 2.2613968047393898
    Converged (small centers change) at step 4/674000


     50%|█████████████████████████████████████████████████████████████████████████                                                                         | 1/2 [00:00<00:00,  1.22it/s]

    Cluster 0 validation auc: 0.753


    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.18it/s]

    Cluster 1 validation auc: 0.758


    
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  4.29it/s]


```python
roc_auc_score(yUnlabeledTest,method2.predict(XUnlabeledTest,bagUnlabeledTest))
```




    0.8493260224267292


